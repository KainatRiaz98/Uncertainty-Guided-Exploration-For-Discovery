import logging
from typing import Tuple

from mlora.model.llm import LlamaModel, LLMModel
from mlora.model.tokenizer import Tokenizer

MODEL_TYPE_DICT = {
    "llama": LlamaModel,
}


def load_partial_model(args) -> LLMModel:
    # load part of model to device
    assert args.rank != -1
    assert len(args.balance) >= args.rank

    logging.info(
        f"Pipeline parallelism, rank is {args.rank} and balance is {args.balance}."
    )

    partial_model_to_device = [
        index + sum(args.balance[: args.rank])
        for index in range(0, args.balance[args.rank])
    ]

    return MODEL_TYPE_DICT[args.model_type].from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
        partial_model_to_device=partial_model_to_device,
    )


def load_full_model(args) -> LLMModel:
    # `devices` / `gpu_layer_balance` are optional and only set by the UG-TTT
    # trainer when a run spans several GPUs. getattr keeps mLoRA's own CLIs
    # (mlora_train.py, mlora_pp_train.py) working unchanged.
    devices = getattr(args, "devices", None)
    layer_balance = getattr(args, "gpu_layer_balance", None)

    if devices:
        logging.info(f"Loading model sharded across devices: {devices}")

    return MODEL_TYPE_DICT[args.model_type].from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
        partial_model_to_device=None,
        devices=devices,
        layer_balance=layer_balance,
    )


def load_model(args) -> Tuple[Tokenizer, LLMModel]:
    assert args.precision in ["nf4", "fp4", "int8", "bf16", "fp16", "fp32"]

    assert args.model_type in MODEL_TYPE_DICT, f"unkown model type {args.model_type}"

    tokenizer = Tokenizer(args.base_model)

    if args.pipeline:
        model = load_partial_model(args)
    else:
        model = load_full_model(args)

    return tokenizer, model
