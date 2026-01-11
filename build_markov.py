from __future__ import annotations
import argparse
import glob
import os
from typing import List
from markov import MarkovChain

def read_corpus(paths: List[str]) -> str:
    parts = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                parts.append(f.read())
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser(description="Build and save a Markov model from corpus files.")
    ap.add_argument("--corpus", nargs="+", required=True, help="Paths/globs to plain-text corpus files")
    ap.add_argument("--order", type=int, default=3, help="Markov chain order (1-3 recommended)")
    ap.add_argument("--out", required=True, help="Output model path (e.g., models/corpus-order3.json.gz)")
    args = ap.parse_args()

    # Resolve corpus globs
    files = []
    for spec in args.corpus:
        files.extend(glob.glob(spec))
    if not files:
        raise SystemExit("No corpus files found. Provide --corpus paths/globs to .txt files.")

    text = read_corpus(files)
    mc = MarkovChain(order=args.order)
    mc.add_text(text)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mc.save(args.out)
    print(f"Saved Markov model: {args.out}")

if __name__ == "__main__":
    main()