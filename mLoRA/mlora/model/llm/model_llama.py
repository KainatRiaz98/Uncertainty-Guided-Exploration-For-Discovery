import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from typing_extensions import override

import torch
from torch.nn.modules import Sequential
from transformers import AutoConfig, AutoModelForCausalLM

from mlora.model.args import LinearInfo, LLMModelArgs, Masks, ModelData
from mlora.model.checkpoint import CheckpointRecomputeFunction
from mlora.model.modules import AdapterModel, Decoder, Embedding, OutputLayer, RMSNorm
from mlora.profiler import nvtx_wrapper, set_backward_tracepoint
from mlora.utils import is_package_available

if is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from mlora.utils import BitsAndBytesConfig

from .model_llm import LLMModel


# input_tokens shape is: batch_size * seq_len
#   default: upper triangular matrix like below, i.e. diagonal = 1
#            0 -inf -inf
#            0    0 -inf
#            0    0    0
# additional_mask: batch_size * seq_len
#   default: is None the matrix like default, if set true, the mask metric will be -inf
#   example: [[True, False, False]]
#           -inf -inf -inf
#           -inf    0 -inf
#           -inf    0    0
def precompute_mask(
    input_tokens: torch.Tensor,
    n_heads: int,
    device: str,
    additional_mask: List[Masks] | None = None,
    diagonal: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if input_tokens.dim() == 2:
        batch_size, seq_len = input_tokens.shape
    elif input_tokens.dim() == 3:
        batch_size, seq_len, _ = input_tokens.shape
    else:
        raise Exception("input dim is not correct {input_tokens.dim}")

    TORCH_MIN_VALUE = torch.finfo(dtype).min
    mask = torch.full(
        (batch_size, n_heads, seq_len, seq_len),
        TORCH_MIN_VALUE,
        device=device,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=diagonal)

    if additional_mask is not None:
        masks_metric = torch.tensor(additional_mask, dtype=torch.bool, device=device)
        masks_metric = masks_metric.view(batch_size, 1, 1, seq_len)
        masks_metric = masks_metric.expand(-1, n_heads, seq_len, -1)
        mask = torch.masked_fill(mask, masks_metric, TORCH_MIN_VALUE)

    mask.requires_grad_(False)

    return mask.to(device=device, dtype=dtype)


LlamaSequentialModuleIO = Tuple[
    torch.Tensor,  # the input batch tokens
    torch.Tensor,  # the mask matrics
    ModelData,  # batch data config
    bool,  # whether to use checkpoint
]
LEN_LLAMA_SEQUENTIAL_MODULE_IO = 4

LlamaCompatibleModelTypes = ["mistral", "qwen2", "qwen3", "llama"]


# ═══════════════════════════════════════════════════════════════════════════
# Multi-GPU layer sharding
#
# The model is split *by layer* across the visible GPUs and executed
# sequentially, hopping the hidden state device->device between layers. This
# adds only device transfers: the ops, their order and their reductions are
# unchanged, so the numbers stay comparable with a single-GPU run.
# ═══════════════════════════════════════════════════════════════════════════


def build_layer_balance(num_layers: int, num_devices: int) -> List[int]:
    """
    Default layer-count-per-device split.

    Remainder layers go to the EARLIER devices on purpose: the last device
    additionally carries model.norm, the lm_head and every fp32 logit/MI
    buffer ((K, chunk, V) fp32 is several GB), so it needs the most headroom.
    """
    assert num_devices > 0, "need at least one device"
    assert num_layers >= num_devices, (
        f"cannot split {num_layers} layers across {num_devices} devices"
    )
    base, rem = divmod(num_layers, num_devices)
    return [base + (1 if i < rem else 0) for i in range(num_devices)]


def build_shard_layout(
    num_layers: int,
    devices: List[str],
    balance: Optional[List[int]] = None,
) -> Tuple[List[str], str, str]:
    """
    Returns (layer_devices, input_device, output_device).

    layer_devices[i] is the device string for decoder layer i.
    input_device holds the embedding; output_device holds norm + lm_head and is
    always the device of the LAST decoder layer, so that the hidden state
    handed to the LM head never has to hop again (this is what lets
    LoRAEnsemble._chunked_logprobs stay untouched).
    """
    if balance is None:
        balance = build_layer_balance(num_layers, len(devices))

    if len(balance) != len(devices):
        raise ValueError(
            f"gpu_layer_balance has {len(balance)} entries but {len(devices)} "
            f"devices were given: {balance} vs {devices}"
        )
    if sum(balance) != num_layers:
        raise ValueError(
            f"gpu_layer_balance sums to {sum(balance)} but the model has "
            f"{num_layers} layers: {balance}"
        )
    if any(b <= 0 for b in balance):
        raise ValueError(f"every device must own at least one layer, got {balance}")

    layer_devices: List[str] = []
    for dev, count in zip(devices, balance):
        layer_devices.extend([dev] * count)

    return layer_devices, devices[0], layer_devices[-1]


def build_hf_device_map(
    config,
    layer_devices: List[str],
    input_device: str,
    output_device: str,
) -> Dict[str, str]:
    """
    Build an explicit HuggingFace device_map covering EVERY top-level submodule.

    The key list is derived by instantiating the architecture on the `meta`
    device (zero memory, zero IO) rather than hardcoded, so a module that only
    exists in some transformers versions -- e.g. `model.rotary_emb` on Qwen3 in
    4.56/4.57 -- cannot be silently left unmapped. An unmapped module stays on
    `meta` and fails much later with a confusing error.
    """
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)

    device_map: Dict[str, str] = {}
    for top_name, top_module in skeleton.named_children():
        if not list(top_module.named_children()):
            # A leaf at the top level (e.g. `lm_head`).
            device_map[top_name] = output_device
            continue
        for sub_name, _ in top_module.named_children():
            full = f"{top_name}.{sub_name}"
            if sub_name == "layers":
                for idx, dev in enumerate(layer_devices):
                    device_map[f"{full}.{idx}"] = dev
            elif sub_name == "embed_tokens":
                device_map[full] = input_device
            else:
                # norm, rotary_emb, and anything else that hangs off the trunk.
                device_map[full] = output_device

    del skeleton
    return device_map


def assert_no_offload(hf_device_map: Dict[str, str]) -> None:
    """
    Fail loudly if any module landed on CPU or disk.

    Silent CPU/disk offload does not corrupt results, but it makes a run
    100x slower -- which on a deadline is just as fatal, and far harder to
    notice than a crash.
    """
    offloaded = {
        name: dev
        for name, dev in hf_device_map.items()
        if str(dev) in ("cpu", "disk", "meta")
    }
    if offloaded:
        raise RuntimeError(
            "Model was partially offloaded to CPU/disk/meta, which would make "
            "training unusably slow. Reduce --context_window or --group_size, "
            "or give the run more GPUs.\nOffloaded modules: "
            + ", ".join(f"{k} -> {v}" for k, v in sorted(offloaded.items())[:20])
            + (" ..." if len(offloaded) > 20 else "")
        )


class LlamaSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, device: Optional[str] = None):
        super().__init__()
        self.wrapper_module_ = module
        # Device this stage's weights live on. None => single-GPU path, no hop.
        self.device_: Optional[str] = device
        # Precomputed: forward() compares against this once per layer per decode
        # step, and a 80-layer model decoding 24k tokens does that ~2M times.
        self.torch_device_: Optional[torch.device] = (
            torch.device(device) if device is not None else None
        )

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: LlamaSequentialModuleIO) -> LlamaSequentialModuleIO:
        assert len(input) == LEN_LLAMA_SEQUENTIAL_MODULE_IO
        assert isinstance(input[0], torch.Tensor)
        assert isinstance(input[1], torch.Tensor)
        assert isinstance(input[2], ModelData)
        assert isinstance(input[3], bool)

        # auto catch the input argument
        @nvtx_wrapper("f_embedding")
        def embedding_forward():
            output = self.wrapper_module_.forward(input[0])
            if input[-1]:
                output = output.requires_grad_(True)
            return (output,) + input[1:]

        def decoder_forward():
            if input[-1]:
                output = CheckpointRecomputeFunction(
                    self.wrapper_module_.forward, *input[:-1]
                )
                set_backward_tracepoint(output.grad_fn, "b_checkpoint")
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output,) + input[1:]

        @nvtx_wrapper("f_rmsnorm")
        def rmsnorm_forward():
            output = self.wrapper_module_.forward(input[0])
            set_backward_tracepoint(output.grad_fn, "b_rmsnorm")
            return (output,) + input[1:]

        @nvtx_wrapper("f_output")
        def output_layer_forward():
            output = self.wrapper_module_.forward(input[0])
            set_backward_tracepoint(output.grad_fn, "b_output")
            return (output,) + input[1:]

        forward_func_dict = {
            "Embedding": embedding_forward,
            "Decoder": decoder_forward,
            "RMSNorm": rmsnorm_forward,
            "OutputLayer": output_layer_forward,
        }

        module_name = self.name()
        assert (
            module_name in forward_func_dict
        ), f"error module name {module_name}"

        return forward_func_dict[module_name]()


class LlamaModel(LLMModel):
    seq_module_: torch.nn.Sequential

    def __init__(self, args: LLMModelArgs):
        self.name_or_path_: str = args.name_or_path_
        # sequential model

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device_
        self.n_heads_ = args.n_heads_
        self.dim_ = args.dim_
        self.vocab_size_ = args.vocab_size_

        # Sharding layout. layer_devices_ is None on the single-GPU path.
        self.layer_devices_ = args.layer_devices_
        self.output_device_ = args.output_device_ or args.device_

        # need to set
        self.pad_token_id_ = args.pad_token_id_
        self.eos_token_id_ = -1

    def layer_devices(self) -> Optional[List[str]]:
        """Device string per decoder layer index, or None when not sharded."""
        return self.layer_devices_

    def output_device(self) -> str:
        """Device holding model.norm and the LM head (where logits appear)."""
        return self.output_device_

    def is_sharded(self) -> bool:
        return self.layer_devices_ is not None and len(set(self.layer_devices_)) > 1

    @override
    def forward(self, input: ModelData) -> torch.Tensor:
        # train model or inference model: output is probs
        tokens = torch.tensor(
            input.batch_tokens_, dtype=torch.int64, device=self.device_
        )

        if getattr(input, 'use_flash_causal_', False):
            # 1-element sentinel: signals flash causal attention, avoids O(n²) mask
            mask = torch.empty(1, device=self.device_)
        else:
            if self.is_sharded():
                # The additive mask is (B, n_heads, T, T); copying it at every
                # layer boundary would dwarf the hidden state. Every UG-TTT call
                # path sets use_flash_causal_, so this is a guard, not a limit.
                raise RuntimeError(
                    "Sharded execution requires use_flash_causal_=True. The "
                    "additive-mask path allocates a (B, n_heads, T, T) tensor "
                    "per device and is not supported across shards."
                )
            mask = precompute_mask(tokens, self.n_heads_, self.device_, input.batch_mask_)

        if input.enable_checkpoint_:
            data = (tokens, mask, input, True)
        else:
            data = (tokens, mask, input, False)

        for seq_layer in self.seq_module_:
            # Skip OutputLayer when only hidden states are needed
            if getattr(input, 'return_hidden_states_', False) and seq_layer.name() == "OutputLayer":
                break

            # ── Cross-device hop ──────────────────────────────────────────
            # Done HERE, outside the wrapper, so it is outside
            # CheckpointRecomputeFunction. Putting a .to() inside the
            # checkpointed region would make the recompute pass see different
            # inputs and desync the dropout RNG.
            #
            # Tensor.to() is differentiable and returns `self` when already on
            # the target device, so this is free on the single-GPU path and
            # autograd carries gradients back across the boundary by itself.
            stage_device = seq_layer.torch_device_
            if stage_device is not None:
                hidden, mask_t = data[0], data[1]
                if hidden.device != stage_device:
                    hidden = hidden.to(stage_device)
                if mask_t.device != stage_device:
                    mask_t = mask_t.to(stage_device)
                data = (hidden, mask_t) + data[2:]

            data = seq_layer.forward(data)

        return data[0]

    @override
    @staticmethod
    def from_pretrained(
        path: str,
        device: str,
        precision: str,
        partial_model_to_device: Optional[List[int]] = None,
        devices: Optional[List[str]] = None,
        layer_balance: Optional[List[int]] = None,
    ) -> LLMModel:
        # ── Multi-GPU layer sharding layout (None on the single-GPU path) ──
        shard_layer_devices: Optional[List[str]] = None
        shard_input_device = device
        shard_output_device = device
        sharded = devices is not None and len(devices) > 1

        if sharded:
            shard_config = AutoConfig.from_pretrained(path)
            if getattr(shard_config, "tie_word_embeddings", False):
                raise RuntimeError(
                    f"{path} has tie_word_embeddings=True, so lm_head shares "
                    "storage with model.embed_tokens. Layer sharding places "
                    "those on different GPUs, which HuggingFace cannot express. "
                    "Use an untied checkpoint (Qwen3-8B/14B/32B and "
                    "Qwen2.5-72B-Instruct are all untied) or run on one GPU."
                )
            (
                shard_layer_devices,
                shard_input_device,
                shard_output_device,
            ) = build_shard_layout(
                shard_config.num_hidden_layers, devices, layer_balance
            )
            logging.info(
                "Sharding %d layers across %s (balance=%s), "
                "embedding on %s, norm+lm_head on %s",
                shard_config.num_hidden_layers,
                devices,
                [shard_layer_devices.count(d) for d in devices],
                shard_input_device,
                shard_output_device,
            )

        # create the device map for parallelism
        def create_device_map() -> str | Dict[str, str]:
            device_map: str | Dict[str, str]
            if sharded:
                device_map = build_hf_device_map(
                    shard_config,
                    shard_layer_devices,
                    shard_input_device,
                    shard_output_device,
                )
            elif partial_model_to_device is None:
                device_map = device
            else:
                config = AutoConfig.from_pretrained(path)
                # Be careful, this is hard coded.
                weight_map = [
                    "model.embed_tokens",
                    *[
                        f"model.layers.{layer_id}"
                        for layer_id in range(0, config.num_hidden_layers)
                    ],
                    "model.norm",
                    "lm_head",
                ]
                device_map = {map_item: "disk" for map_item in weight_map}
                for partial_weight in partial_model_to_device:
                    device_map[weight_map[partial_weight]] = device
            return device_map

        # the argument for the LlamaForCausalLM load the pretrained large model
        load_type_dict = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        additional_load_args = {
            "device_map": create_device_map(),
            "torch_dtype": torch.float32,
        }

        logging.info(f"Loading model with precision - {precision}")

        if precision in load_type_dict:
            additional_load_args["torch_dtype"] = load_type_dict[precision]
        else:
            load_4bit = precision in ["nf4", "fp4"]
            load_8bit = precision == "int8"

            additional_load_args["torch_dtype"] = torch.float32
            additional_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                # int8 only for GPU, fp32 for cpu
                llm_int8_enable_fp32_cpu_offload=True,
                # do not hold the fp16 part
                # when forward and backward need to convert int8 to fp16
                llm_int8_has_fp16_weight=False,
                # only for qlora 4bit
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=precision,
            )

        llama_model = AutoModelForCausalLM.from_pretrained(path, **additional_load_args)

        if llama_model.config.model_type not in LlamaCompatibleModelTypes:
            assert False, f"unsupported model type {llama_model.config.model_type}, loading with llama compatible mode."

        logging.info(
            f"loading llama compatible model - {llama_model.config.model_type}"
        )

        if sharded:
            hf_map = getattr(llama_model, "hf_device_map", None)
            if not hf_map:
                raise RuntimeError(
                    "Requested multi-GPU sharding but transformers did not "
                    "produce an hf_device_map. This usually means `accelerate` "
                    "is not installed (`pip install accelerate`)."
                )
            assert_no_offload(hf_map)
            logging.info("hf_device_map resolved to %d entries", len(hf_map))

        llama_args = LLMModelArgs(llama_model.config)
        if llama_args.pad_token_id_ is None:
            llama_args.pad_token_id_ = -1
        llama_args.device_ = shard_input_device
        llama_args.dtype_ = llama_model.dtype
        llama_args.layer_devices_ = shard_layer_devices
        llama_args.output_device_ = shard_output_device

        # RoPE tables are sized by max_seq_len_ and sliced with no bounds check
        # in Attention.forward, so a too-small table fails deep inside
        # apply_rotary_emb rather than here. Surface it at load time instead.
        if llama_args.max_seq_len_ < 8192:
            logging.warning(
                "max_seq_len_ resolved to %d -- RoPE tables are sized to this "
                "and positions beyond it will fail with a shape mismatch inside "
                "apply_rotary_emb. Check the model config's "
                "max_position_embeddings / sliding_window.",
                llama_args.max_seq_len_,
            )

        # load model from pretrained large model
        model = LlamaModel.convert_model_from_huggingface(llama_model, llama_args)

        return model

    @staticmethod
    def convert_model_from_huggingface(
        llama_model: AutoModelForCausalLM, llama_args: LLMModelArgs
    ):
        llama_model.requires_grad_(False)

        # Per-stage device, or None everywhere on the single-GPU path (in which
        # case LlamaModel.forward skips the hop entirely).
        layer_devices = llama_args.layer_devices_
        is_sharded = layer_devices is not None
        in_dev = llama_args.device_ if is_sharded else None
        out_dev = llama_args.output_device_ if is_sharded else None

        seq_model: OrderedDict[str, torch.nn.Module] = OrderedDict()

        seq_model.update(
            {
                "embedding": LlamaSequentialWrapper(
                    Embedding(
                        llama_model.model.embed_tokens.weight, llama_args.pad_token_id_
                    ),
                    device=in_dev,
                )
            }
        )

        for idx, target_layer in enumerate(llama_model.model.layers):
            layer_dev = layer_devices[idx] if is_sharded else None
            # Build the RoPE tables directly on this layer's device. They are
            # plain tensor attributes, not registered buffers, so a later
            # module.to() would NOT move them.
            decoder = Decoder(idx, llama_args, device=layer_dev)
            decoder.from_pretrained(target_layer, llama_args.norm_eps_)
            seq_model.update(
                {f"layer{idx}": LlamaSequentialWrapper(decoder, device=layer_dev)}
            )

        seq_model.update(
            {
                "norm": LlamaSequentialWrapper(
                    RMSNorm(llama_model.model.norm.weight, llama_args.norm_eps_),
                    device=out_dev,
                )
            }
        )
        seq_model.update(
            {
                "output": LlamaSequentialWrapper(
                    OutputLayer(llama_model.lm_head.weight, llama_args),
                    device=out_dev,
                )
            }
        )

        model = LlamaModel(llama_args)
        model.seq_module_ = torch.nn.Sequential(seq_model)

        return model

    @override
    def load_adapter(self, adapter_model: AdapterModel):
        # module is LlamaSequentialWrapper
        for module in self.seq_module_:
            if module.name() != "Decoder":
                continue
            module.wrapper_module_.load_adapter(adapter_model)

    @override
    def offload_adapter(self, adapter_name: str):
        # now only transformers block have adapter
        for module in self.seq_module_:
            if module.name() != "Decoder":
                continue
            module.wrapper_module_.offload_adapter(adapter_name)

    @override
    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()
        for module in self.seq_module_:
            if module.name() != "Decoder":
                continue
            ret_val.update(module.wrapper_module_.linears_info())
        return ret_val

    @override
    def sequential(self) -> Sequential:
        return self.seq_module_
