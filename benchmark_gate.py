"""Confronta un checkpoint candidato con il riferimento stabile."""
import argparse
from evaluate_8_questions import evaluate_model

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
    if cand < base:
        raise SystemExit("REJECTED: il candidato peggiora il benchmark stabile")
    if args.require_improvement and cand <= base:
        raise SystemExit("REJECTED: il candidato non migliora il benchmark stabile")
    print("ACCEPTED: il candidato non peggiora il benchmark")

if __name__ == "__main__":
    main()
