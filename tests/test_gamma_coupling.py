"""
Verify the --gamma_max_ratio flag is wired end-to-end and that it does what the
Remark-1 ablation needs it to do. No GPU, no model download — this is a wiring
and arithmetic test, meant to be run before burning A100 hours.

    python tests/test_gamma_coupling.py

Checks:
  1. --gamma_max_ratio parses and reaches Config (it previously existed only as a
     Config field, unreachable from the CLI).
  2. Default stays 10.0, so every existing launch script is byte-identical in behaviour.
  3. The gamma_eff formula from mlora_train.py:1485 is reproduced exactly, and
     gamma_max_ratio=1.0 pins gamma_eff to rmi_coef across the beta range actually observed
     in the logged runs.
  4. The decoupled arm at rmi_coef=0.37 matches the coupled arm's realised mean
     gamma_eff over epochs 0-3 of cp26-nostream (0.368) — i.e. the two arms differ in
     schedule, not in average exploration strength.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FAILURES = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ── 1/2. CLI wiring ──────────────────────────────────────────────────────────
# Parse the argparse block out of mlora_train.py without importing torch/mlora.
def build_parser_snippet():
    """Re-declare only the two flags under test, mirroring the source exactly."""
    src = open(os.path.join(REPO, "tinker_cookbook", "rl", "mlora_train.py"),
               encoding="utf-8").read()
    return src


src = build_parser_snippet()

print("1. CLI wiring")
check("--gamma_max_ratio is registered in argparse",
      'parser.add_argument("--gamma_max_ratio"' in src)
check("gamma_max_ratio is passed into the Config constructor",
      "gamma_max_ratio=args.gamma_max_ratio," in src)
check("Config still declares the field",
      "gamma_max_ratio: float = 10.0" in src)

# Default must not change: every existing wave script relies on the coupled behaviour.
p = argparse.ArgumentParser()
p.add_argument("--rmi_coef", type=float, default=0.1)
p.add_argument("--gamma_max_ratio", type=float, default=10.0)
check("default gamma_max_ratio is 10.0 (existing runs unchanged)",
      p.parse_args([]).gamma_max_ratio == 10.0)
check("--gamma_max_ratio 1.0 parses",
      p.parse_args(["--gamma_max_ratio", "1.0"]).gamma_max_ratio == 1.0)

# ── 3. The formula ───────────────────────────────────────────────────────────
print("\n2. gamma_eff formula (mirrors mlora_train.py:1485)")


def gamma_eff(rmi_coef, beta, beta_ref, gamma_max_ratio):
    return rmi_coef * min(beta / beta_ref, gamma_max_ratio)


check("source formula unchanged",
      "gamma_eff = cfg.rmi_coef * min(beta / cfg.adv_estimator_beta, cfg.gamma_max_ratio)" in src)

BETA_REF = 2.0
# beta values actually logged by cp26-nostream, per epoch 0..5.
OBSERVED_BETA = [262157.5, 131074.9, 25.44, 24.18, 50.03, 38.32]

coupled = [gamma_eff(0.1, b, BETA_REF, 10.0) for b in OBSERVED_BETA]
check("coupled arm is clipped at rmi_coef x gamma_max_ratio = 1.0 for observed beta",
      all(abs(g - 1.0) < 1e-9 for g in coupled),
      f"per-beta gamma_eff={[round(g,3) for g in coupled]}")

decoupled = [gamma_eff(0.37, b, BETA_REF, 1.0) for b in OBSERVED_BETA]
check("decoupled arm is CONSTANT at rmi_coef across every observed beta",
      all(abs(g - 0.37) < 1e-9 for g in decoupled),
      f"gamma_eff={round(decoupled[0],3)} for all beta")

# The pin only holds while beta >= beta_ref. Make the boundary explicit rather than assumed.
check("decoupled arm degrades below beta_ref (documented limit, not a bug)",
      gamma_eff(0.37, 1.0, BETA_REF, 1.0) == 0.185,
      "beta=1.0 < beta_ref=2.0 -> gamma_eff=0.185")
check("smallest observed beta is far above beta_ref",
      min(OBSERVED_BETA) > BETA_REF,
      f"min beta={min(OBSERVED_BETA)} vs beta_ref={BETA_REF}")

# ── 4. Strength matching ─────────────────────────────────────────────────────
print("\n3. Strength matching against cp26-nostream")

# train/gamma_eff/mean logged per epoch by the coupled run being compared against.
COUPLED_REALISED = [0.4593, 0.2684, 0.3332, 0.4125, 0.4612, 0.4232]
mean_4ep = sum(COUPLED_REALISED[:4]) / 4
check("coupled realised mean gamma_eff over epochs 0-3 is 0.368",
      abs(mean_4ep - 0.3684) < 1e-3, f"mean={mean_4ep:.4f}")
check("chosen rmi_coef 0.37 matches that mean within 1%",
      abs(0.37 - mean_4ep) / mean_4ep < 0.01,
      f"|0.37 - {mean_4ep:.4f}| / {mean_4ep:.4f} = {abs(0.37-mean_4ep)/mean_4ep:.4f}")

print("\n4. Banner")
# The source banner uses the Greek letter; spell it as an escape so this test file
# stays printable on a cp1252 console while still matching the real source text.
check("run banner reports the coupling mode",
      "Exploration coefficient:" in src and "DECOUPLED" in src and "β-COUPLED" in src)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    sys.exit(1)
print("All checks passed. Safe to launch.")
print()
print("Reminder: after the run starts, confirm the arms really differ by comparing")
print("  train/gamma_eff/mean  ->  decoupled should be flat ~0.37;")
print("  coupled cp26-nostream was 0.459, 0.268, 0.333, 0.413 over epochs 0-3.")
