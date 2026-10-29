"""Confronta un checkpoint candidato con il riferimento stabile."""
import argparse
from evaluate_8_questions import evaluate_model

def accept_candidate(baseline: int, candidate: int, require_improvement: bool = False) -> bool:
    """Restituisce True solo se il candidato rispetta il gate scelto."""
    if candidate < baseline:
        return False
    return not require_improvement or candidate > baseline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("--baseline", default="checkpoints_lumina_next")
    p.add_argument("--require-improvement", action="store_true",
                   help="accetta solo se il candidato supera il baseline")
    args = p.parse_args()
    base, _ = evaluate_model(args.baseline, "gate_baseline")
    cand, _ = evaluate_model(args.candidate, "gate_candidate")
    print(f"baseline={base}/16 candidate={cand}/16")
    if not accept_candidate(base, cand, args.require_improvement):
        if cand < base:
            raise SystemExit("REJECTED: il candidato peggiora il benchmark stabile")
        raise SystemExit("REJECTED: il candidato non migliora il benchmark stabile")
    print("ACCEPTED: il candidato non peggiora il benchmark")

if __name__ == "__main__":
    main()
