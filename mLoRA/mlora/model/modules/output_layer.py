import torch

from mlora.model.args import LLMModelArgs


class OutputLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, args: LLMModelArgs):
        super().__init__()

        # Under layer sharding the LM head lives on the LAST decoder layer's
        # GPU, so the hidden state handed to it never has to hop again. Falls
        # back to args.device_ on the single-GPU path.
        target_device = args.output_device_ or args.device_

        def _as_param(w: torch.Tensor) -> torch.nn.Parameter:
            if isinstance(w, torch.nn.Parameter):
                return w
            return torch.nn.Parameter(w, requires_grad=False)

        already_placed = (
            weight.device != torch.device("meta")
            and torch.device(weight.device) == torch.device(target_device)
            and weight.dtype == args.dtype_
            and tuple(weight.shape) == (args.vocab_size_, args.dim_)
        )

        if weight.device == torch.device("meta") or already_placed:
            # Reference the existing weight instead of allocating a second copy.
            # For Qwen2.5-72B that duplicate is ~2.5 GB (fp16) / ~5 GB (fp32) on
            # a single GPU. Referencing is numerically identical: the base model
            # is frozen (requires_grad_(False)) and nothing mutates lm_head.
            self.lm_head_ = torch.nn.Linear(
                args.dim_,
                args.vocab_size_,
                bias=False,
                device="meta",
                dtype=args.dtype_,
            )
            self.lm_head_.weight = _as_param(weight)
        else:
            # Device or dtype mismatch: allocate on the target and copy across.
            # Tensor.copy_ handles the cross-device case and is exact.
            self.lm_head_ = torch.nn.Linear(
                args.dim_,
                args.vocab_size_,
                bias=False,
                device=target_device,
                dtype=args.dtype_,
            )
            with torch.no_grad():
                self.lm_head_.weight.copy_(weight)

        self.lm_head_.requires_grad_(False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()
