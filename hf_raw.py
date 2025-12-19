from __future__ import annotations
import argparse
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from markov import MarkovChain

def read_corpus(paths):
    import os
    parts = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                parts.append(f.read())
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser(description="Raw continuation with HF Transformers (no instruction prompt).")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="HF model id (base/causal LM preferred)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--seed-len", type=int, default=12)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rng-seed", type=int, default=None)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if args.device=="cuda" else torch.float32)
    model.to(device)
    model.eval()

    from random import Random
    rng = Random(args.rng_seed)
    text = read_corpus([p for spec in args.corpus for p in glob.glob(spec)])
    mc = MarkovChain(order=args.order)
    mc.add_text(text)

    for i in range(args.count):
        seed_words = mc.generate_seed_text(n_words=args.seed_len, rng=rng)
        context = f"{args.prefix} {seed_words}".strip() if args.prefix else seed_words
        inputs = tokenizer(context, return_tensors="pt").to(device)
        out_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
        full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # Print only the new continuation beyond the context:
        print(full[len(context):].strip())
        print("---")

if __name__ == "__main__":
    main()