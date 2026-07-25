#!/usr/bin/env python3
"""
Acceptance gate for multi-GPU layer sharding.

Proves that running a model split across N GPUs produces the same numbers as
running it on one GPU. Run this on a model that FITS on one GPU (Qwen3-8B, or
Qwen3-1.7B for fast iteration) so both sides are directly comparable, then trust
the same code path for 14B/32B/72B.

Two passes, because holding both a single-GPU and a sharded copy at once would
double the memory:

    export PYTHONPATH=mLoRA
    python tests/test_sharded_equivalence.py --model Qwen/Qwen3-8B --gpus 1 --out /tmp/ref.pt
    python tests/test_sharded_equivalence.py --model Qwen/Qwen3-8B --gpus 2 --out /tmp/shard.pt
    python tests/test_sharded_equivalence.py --compare /tmp/ref.pt /tmp/shard.pt

WHY THE DROPOUT SWITCH IS NEEDED
--------------------------------
LoRAFunction.forward calls F.dropout without passing `training`, so it defaults
to True and stays active under torch.no_grad() and after seq_module_.eval() --
eval() cannot reach a functional call. Every logprob and MI value is therefore a
draw from a random variable, and under sharding each layer draws from a
DIFFERENT GPU's RNG stream. Without UGTTT_DISABLE_LORA_DROPOUT=1 (set below,
before any mlora import) the two sides can never agree and this test is
meaningless. Production runs leave it unset.
"""

import os

# MUST be set before mlora.model.modules.lora is imported — it is read at
# module import time.
os.environ.setdefault("UGTTT_DISABLE_LORA_DROPOUT", "1")

import argparse
import hashlib
import sys
from typing import Any, Dict, List

import torch


# Tolerances. fp16 weights with an fp32 LM head: cross-device copies are exact,
# so any disagreement above these comes from kernel selection, not from the
# sharding itself. A REAL bug looks structured (one layer boundary, one adapter,
# or error growing with depth), not like uniform small noise.
TOL_LOGPROB = 1e-3     # fp32 LM head output
TOL_MI = 1e-3          # fp32 entropy arithmetic
TOL_WEIGHT = 1e-5      # fp32 LoRA weights after one AdamW step


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()[:16]


def build(model_name: str, gpus: str, balance: str, k: int, rank: int, seed: int):
    from mlora.model import load_model as mlora_load_model
    from tinker_cookbook.rl.ensemble import LoRAEnsemble
    from tinker_cookbook.rl.mlora_train import (
        parse_layer_balance,
        resolve_shard_devices,
        seed_everything,
    )

    seed_everything(seed)
    devices = resolve_shard_devices(gpus)

    args = argparse.Namespace(
        base_model=model_name,
        device=devices[0] if devices else "cuda",
        precision="fp16",
        model_type="llama",
        pipeline=False,
        balance=None,
        rank=None,
        devices=devices,
        gpu_layer_balance=parse_layer_balance(balance),
    )
    tokenizer, model = mlora_load_model(args)

    ensemble = LoRAEnsemble(
        model=model,
        num_members=k,
        lora_rank=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        learning_rate=4e-5,
        optimizer="adamw",
        seed=seed,
        # Forced ON on both sides so the two passes use the same code path;
        # otherwise the sharded side would differ purely by its default.
        memory_efficient_prefill=True,
    )
    return tokenizer, model, ensemble


def collect(args) -> Dict[str, Any]:
    from tinker_cookbook.rl.mlora_train import compute_rl_loss, _compute_base_logprobs
    from tinker_cookbook.rl.nuclear_norm import compute_nuclear_norm_diversity_loss
    from tinker_cookbook.rl.uncertainty import compute_true_mi

    tokenizer, model, ensemble = build(
        args.model, args.gpus, args.gpu_layer_balance, args.k, args.rank, args.seed
    )
    out: Dict[str, Any] = {"config": vars(args)}

    # ── 0. Placement ──────────────────────────────────────────────────────
    layer_devices = getattr(model, "layer_devices", lambda: None)()
    out["placement"] = {
        "sharded": bool(getattr(model, "is_sharded", lambda: False)()),
        "layer_devices": layer_devices,
        "output_device": str(getattr(model, "output_device", lambda: "?")()),
        "logits_device": str(ensemble.logits_device()),
        "n_layers": len(layer_devices) if layer_devices else None,
    }
    if layer_devices:
        assert str(ensemble.logits_device()) == str(layer_devices[-1]), (
            f"lm_head is on {ensemble.logits_device()} but the last decoder "
            f"layer is on {layer_devices[-1]} — _chunked_logprobs would hop."
        )
        for ki, ctx in enumerate(ensemble.contexts):
            for tname, module in ctx.adapter_model_.items():
                want = torch.device(layer_devices[int(tname.split(".")[1])])
                for ab in ("a", "b"):
                    got = getattr(module, f"lora_{ab}_").device
                    assert got == want, (
                        f"adapter {ki}:{tname}.lora_{ab} is on {got} but its "
                        f"layer is on {want}"
                    )
    print(f"  placement ok: {out['placement']['output_device']=} "
          f"{out['placement']['logits_device']=}")

    # ── 1. Adapter init — must be BITWISE identical ───────────────────────
    # lora_a_/lora_b_ are created AND initialised on CPU before any device
    # move, so sharding cannot touch them. A mismatch here means placement
    # leaked into initialisation.
    out["init_hashes"] = {
        f"{ki}:{name}:{ab}": sha(getattr(mod, f"lora_{ab}_"))
        for ki, ctx in enumerate(ensemble.contexts)
        for name, mod in ctx.adapter_model_.items()
        for ab in ("a", "b")
    }
    print(f"  hashed {len(out['init_hashes'])} adapter tensors")

    # ── 2. Deterministic forward: logprobs + MI ───────────────────────────
    torch.manual_seed(args.seed)
    tokens: List[int] = tokenizer.encode(
        "def solve(n):\n    # find the optimal packing\n    total = 0\n"
        "    for i in range(n):\n        total += i * i\n    return total\n" * 6,
        bos=True, eos=False,
    )
    tokens = tokens[: args.seq_len]
    out["tokens"] = tokens

    logprobs, per_token_mi = ensemble.compute_ensemble_logprobs(tokens)
    out["logprobs"] = logprobs.detach().cpu()
    out["per_token_mi"] = per_token_mi.detach().cpu()
    out["true_mi"] = float(compute_true_mi(per_token_mi).item())
    print(f"  logprobs {tuple(logprobs.shape)}  true_mi={out['true_mi']:.6f}")

    assert torch.isfinite(logprobs).all(), "non-finite logprobs"
    assert torch.isfinite(per_token_mi).all(), "non-finite MI"
    assert out["true_mi"] > 0, "MI is zero — the ensemble has collapsed"

    # ── 3. Base logprobs (the KL-penalty path, chunked LM head) ───────────
    out["base_logprobs"] = _compute_base_logprobs(ensemble, tokens).detach().cpu()

    # ── 4. Greedy generation must match EXACTLY ───────────────────────────
    # temperature=0 takes the argmax branch, so no RNG is consumed by sampling.
    # This isolates "is the sharded forward correct" from "where does the RNG
    # live", which is the one thing sharding legitimately changes.
    torch.manual_seed(args.seed)
    gen_tokens, gen_logprobs = ensemble.generate(
        prompt_tokens=tokens[: min(len(tokens), 64)],
        max_tokens=args.gen_tokens,
        temperature=0.0,
        eos_token_id=tokenizer.eos_id_,
        adapter_idx=0,
    )
    out["greedy_tokens"] = list(gen_tokens)
    out["greedy_logprobs"] = list(gen_logprobs)
    print(f"  greedy generated {len(out['greedy_tokens'])} tokens")

    # ── 5. One training step ──────────────────────────────────────────────
    # Fixed advantage and mask so the only variable is the forward/backward.
    torch.manual_seed(args.seed)
    dev = ensemble.logits_device()
    action_mask = torch.ones(len(tokens) - 1, device=dev)
    old_logprobs = logprobs[0].detach().clone()

    for ki in range(ensemble.K):
        _hidden, lp_k = ensemble.compute_training_logprobs_single(ki, tokens)
        del _hidden
        loss = compute_rl_loss(
            current_logprobs=lp_k,
            old_logprobs=old_logprobs,
            advantage=0.5,
            action_mask=action_mask,
            loss_fn="importance_sampling",
        )
        (loss / ensemble.K).backward()
        del lp_k, loss

    # nnm exercises the cross-device accumulator; it is ON in the flagship arm.
    nnm_loss, nnm_nuc, _ = compute_nuclear_norm_diversity_loss(ensemble)
    (0.075 * nnm_loss).backward()
    out["nnm_nuclear_norm"] = float(nnm_nuc)

    ensemble.step_all_optimizers()

    out["post_step_hashes"] = {
        f"{ki}:{name}:{ab}": sha(getattr(mod, f"lora_{ab}_"))
        for ki, ctx in enumerate(ensemble.contexts)
        for name, mod in ctx.adapter_model_.items()
        for ab in ("a", "b")
    }
    out["post_step_sample"] = {
        f"{ki}:{ab}": getattr(
            list(ensemble.contexts[ki].adapter_model_.values())[0], f"lora_{ab}_"
        ).detach().cpu()
        for ki in range(ensemble.K)
        for ab in ("a", "b")
    }
    print(f"  train step ok, nnm_nuclear_norm={nnm_nuc:.6f}")

    from tinker_cookbook.rl.gpu_utils import memory_metrics
    out["memory"] = memory_metrics()
    return out


def compare(ref_path: str, shard_path: str) -> int:
    ref = torch.load(ref_path, weights_only=False, map_location="cpu")
    sh = torch.load(shard_path, weights_only=False, map_location="cpu")
    failures: List[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
        if not cond:
            failures.append(label)

    def close(label: str, a: torch.Tensor, b: torch.Tensor, tol: float) -> None:
        if a.shape != b.shape:
            check(label, False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
            return
        d = (a.float() - b.float()).abs()
        check(label, bool(d.max() <= tol),
              f"max|d|={d.max():.3e} mean|d|={d.mean():.3e} tol={tol:.0e}")

    print(f"\nreference : {ref['placement']}")
    print(f"sharded   : {sh['placement']}\n")

    check("sharded run actually sharded", sh["placement"]["sharded"] is True)
    check("same prompt tokens", ref["tokens"] == sh["tokens"])

    # 1. Bitwise — this one has no tolerance. Adapter init happens on CPU.
    mism = [k for k in ref["init_hashes"] if ref["init_hashes"][k] != sh["init_hashes"].get(k)]
    check("adapter init BITWISE identical", not mism,
          "" if not mism else f"{len(mism)}/{len(ref['init_hashes'])} differ, e.g. {mism[:3]}")

    # 2-3. Numerics
    close("per-token logprobs", ref["logprobs"], sh["logprobs"], TOL_LOGPROB)
    close("per-token MI", ref["per_token_mi"], sh["per_token_mi"], TOL_MI)
    close("base logprobs (KL path)", ref["base_logprobs"], sh["base_logprobs"], TOL_LOGPROB)
    check("top-7% true MI", abs(ref["true_mi"] - sh["true_mi"]) <= TOL_MI,
          f"{ref['true_mi']:.6f} vs {sh['true_mi']:.6f}")

    # 4. Greedy decode — exact, no tolerance.
    same = ref["greedy_tokens"] == sh["greedy_tokens"]
    if not same:
        n = min(len(ref["greedy_tokens"]), len(sh["greedy_tokens"]))
        first = next((i for i in range(n)
                      if ref["greedy_tokens"][i] != sh["greedy_tokens"][i]), n)
        check("greedy decode identical", False, f"first divergence at token {first}")
    else:
        check("greedy decode identical", True, f"{len(ref['greedy_tokens'])} tokens")

    if ref.get("greedy_logprobs") and sh.get("greedy_logprobs"):
        close("greedy logprobs",
              torch.tensor(ref["greedy_logprobs"]),
              torch.tensor(sh["greedy_logprobs"]), TOL_LOGPROB)

    # 5. Training
    check("nnm nuclear norm",
          abs(ref["nnm_nuclear_norm"] - sh["nnm_nuclear_norm"]) <= 1e-3,
          f"{ref['nnm_nuclear_norm']:.6f} vs {sh['nnm_nuclear_norm']:.6f}")
    for key in sorted(ref["post_step_sample"]):
        close(f"post-step lora_{key}", ref["post_step_sample"][key],
              sh["post_step_sample"][key], TOL_WEIGHT)

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        print("\nDo NOT launch big-model runs. A structured divergence (one layer\n"
              "boundary, one adapter, or error growing with depth) means a real\n"
              "placement bug; uniform small noise above tolerance usually means\n"
              "the tolerance needs justifying, not loosening.")
        return 1
    print("ALL CHECKS PASSED — sharded execution matches single-GPU.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("REF", "SHARD"))
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--gpus", default="1")
    p.add_argument("--gpu_layer_balance", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=32)
    args = p.parse_args()

    if args.compare:
        return compare(*args.compare)

    if not args.out:
        p.error("--out is required unless --compare is given")

    print(f"Collecting fingerprints: model={args.model} gpus={args.gpus}")
    result = collect(args)
    torch.save(result, args.out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
