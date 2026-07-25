# Running 14B / 32B / 72B: multi-GPU layer sharding

Branch: `feat/multi-gpu-sharding`. Target: one 8×80GB node.

The trainer previously could only load a model onto a single GPU. It now splits
the layer stack across GPUs and runs it sequentially, hopping the hidden state
device→device between stages. Nothing about the maths changes — same ops, same
order, same reductions — so results stay comparable with the 8B runs.

`--precision fp16` at **every** model size. Sharding is what makes that
possible, and it means model size is the only variable across the scaling curve
(the earlier plan used nf4 for 72B, which confounded size with quantization).

---

## Do these in order

### 0. Check the environment

```bash
python -c "import accelerate, transformers, torch; print(accelerate.__version__, transformers.__version__, torch.__version__)"
```

`accelerate` is **required** for any `device_map` and is not pinned in
`requirements/*.txt`. It is already present on the running boxes (the current
code passes `device_map="cuda"`, which also needs it), but check before you
start — without it the load fails with a confusing error.

### 1. Pre-flight — config only, no weights downloaded

```bash
python scripts/preflight_model_check.py --models Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B Qwen/Qwen2.5-72B-Instruct --gpu-memory-gb 80 --num-gpus 8
```

Reports per-model memory and the `--gpus` to use. It also checks the things that
otherwise fail deep inside `apply_rotary_emb` rather than at load: `model_type`
compatibility, `tie_word_embeddings`, `rope_scaling` (which `LLMModelArgs`
silently drops), and `max_seq_len_ ≥ context_window`. Non-zero exit = do not
proceed.

### 2. Acceptance gate — proves sharded == single-GPU

Run on a model that fits on one GPU, so both sides are directly comparable.

```bash
export PYTHONPATH=mLoRA
python tests/test_sharded_equivalence.py --model Qwen/Qwen3-8B --gpus 1 --out /tmp/ref.pt
python tests/test_sharded_equivalence.py --model Qwen/Qwen3-8B --gpus 2 --out /tmp/shard.pt
python tests/test_sharded_equivalence.py --compare /tmp/ref.pt /tmp/shard.pt
```

Use `Qwen/Qwen3-1.7B` first if you want a 2-minute loop.

Two checks have **no tolerance** and are the ones that matter most:

- **Adapter init must be bitwise identical.** `lora_a_`/`lora_b_` are created and
  initialised on CPU before any device move, so sharding cannot touch them. A
  mismatch means placement leaked into initialisation.
- **Greedy decode (`temperature=0`) must match exactly.** The argmax branch
  consumes no RNG, so this isolates "is the sharded forward correct" from "where
  does the RNG live".

For logprobs/MI, a real bug looks *structured* — one layer boundary, one
adapter, or error growing with depth. Uniform small noise above tolerance means
the tolerance needs justifying, not loosening.

### 3. Smoke the largest model before committing the box

```bash
bash scripts/aws/launch_scaling_wave.sh smoke72
```

### 4. Launch

```bash
bash scripts/aws/launch_scaling_wave.sh --list
bash scripts/aws/launch_scaling_wave.sh scale72
```

---

## GPU budget

Pre-allocated KV is **1.0 GiB per layer** at `context_window=32768`,
`group_size=8` — all three models have `n_kv_heads × head_dim = 1024`, so it
depends only on layer count. It dominates, and it is **linear** in
`context_window` and `group_size`.

| Model | Layers | fp16 weights | KV | Total | GPUs @ 80GB |
|---|---|---|---|---|---|
| Qwen3-14B | 40 | ~29 GB | 40 GB | ~73 GB | 2 (1 is marginal) |
| Qwen3-32B | 64 | ~65 GB | 64 GB | ~133 GB | 2 tight, 3 comfortable |
| Qwen2.5-72B | 80 | ~145 GB | 80 GB | ~231 GB | 4 |

If a run OOMs, prefer giving it another GPU over shrinking `context_window` or
`group_size` — those change the experiment.

If the **last** GPU is the one that OOMs, that is expected: it also carries
`model.norm`, the LM head, and the fp32 `(K, chunk, V)` MI buffers. Shift a
couple of layers off it:

```bash
--gpus 4 --gpu_layer_balance 21,21,21,17
```

Waves: `smoke72` (1×4 GPUs), `scale72` (2×4), `scale_small` (4×2, 32B tight),
`scale32safe` (2×3 + 1×2).

---

## Reproducibility — state this honestly in the rebuttal

Sharding changes **where** randomness is drawn: LoRA dropout now draws from each
layer's own GPU RNG stream, and `torch.multinomial` draws on the last GPU
instead of `cuda:0`. So a sharded run is **not** token-identical to a
single-GPU run of the same model — it is a different, equally valid draw.

It **is** fully reproducible for a fixed `(seed, shard layout)`, and adapter
initialisation is identical either way because it happens on CPU. Acceptance
tests 1 and 4 are what demonstrate the difference is RNG placement, not a
numerical defect.

Record `--gpus` and `--gpu_layer_balance` alongside the seed for every reported
run. Checkpoints are layout-independent (`map_location="cpu"`), so a run can be
resumed under a different split.

---

## Red flags in the logs

| Signature | Meaning |
|---|---|
| `PROMPT TRUNCATED` | Prompt exceeded the 4096-token cap. See below. |
| `partially offloaded` | A shard landed on CPU/disk; the run would be ~100× slower. Load aborts. |
| `uncertainty_true_mi` = 0 | Ensemble collapsed — MI carries no signal. |
| `uncertainty_true_mi` = NaN | Numerical failure in the MI kernel. |
| `reward_exec` all zero for an epoch | Verifier or code-extraction failure, not a model result. |
| Last GPU peak within ~2 GB of capacity | Rebalance before it OOMs mid-run. |

Check placement actually happened:

```bash
grep -E "Shard layout|logits land on" logs/aws/<wave>/*.log
```

---

## Known pre-existing issue: the 4096-token prompt cap

`Tokenizer.encode` has `cutoff_len=4096` as a default and
[mlora_train.py:1853](../tinker_cookbook/rl/mlora_train.py:1853) never overrides
it. **Behaviour is unchanged on this branch** — only a warning was added.

It matters because in the prompt templates the `<<<LAST_CODE>>>` placeholder
sits near the *end*. What truncation removes first is the value context, the
Rules, and the line *"return the final program between \`\`\`python and
\`\`\`"* — and without that, `last_codeblock_postprocess` finds no fenced block
and the rollout scores exactly `0.0`. The symptom is a silent reward collapse
that switches on once the carried-forward program gets long enough.

Budget for the previous program is roughly **3,200 tokens (~10 KB)** after the
template (AC1 2,513 chars / AC2 2,686 / CP 1,570). The math domains
(ac1/ac2/cp/erdos) carry compact programs and stay under it. The long-program
domains do not — the real artifacts in this repo are `ahc039.cpp` 35.5 KB,
`ahc058.cpp` 22.7 KB, `mla_code_*.py` 23–27 KB, i.e. 2–3× over.

If `PROMPT TRUNCATED` appears in a scaling run, that run's rewards are not
measuring what they claim to.
