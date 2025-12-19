from __future__ import annotations
import re
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence

_word_re = re.compile(r"[A-Za-zÀ-ÿ']+|[.?!]")

def tokenize(text: str) -> List[str]:
    # Simple tokenization: words and sentence punctuation
    return _word_re.findall(text)

class MarkovChain:
    def __init__(self, order: int = 2):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.transitions: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        self.starts: List[Tuple[str, ...]] = []

    def add_text(self, text: str):
        tokens = tokenize(text)
        if len(tokens) <= self.order:
            return
        # naive sentence boundary detection by punctuation
        sent_starts = [0]
        for i, tok in enumerate(tokens):
            if tok in (".", "?", "!") and i + 1 < len(tokens):
                sent_starts.append(i + 1)

        for s in sent_starts:
            if s + self.order < len(tokens):
                self.starts.append(tuple(tokens[s : s + self.order]))

        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            nxt = tokens[i + self.order]
            self.transitions[state].append(nxt)

    def _next(self, state: Tuple[str, ...], rng: random.Random) -> str | None:
        candidates = self.transitions.get(state)
        if not candidates:
            return None
        return rng.choice(candidates)

    def generate_words(self, n_words: int, rng: random.Random) -> List[str]:
        if not self.starts:
            return []
        state = rng.choice(self.starts)
        out = list(state)
        while len([t for t in out if t.isalpha()]) < n_words:
            nxt = self._next(tuple(out[-self.order:]), rng)
            if nxt is None:
                # restart from a random start
                state = rng.choice(self.starts)
                out.extend(state)
                continue
            out.append(nxt)
        # Strip trailing punctuation noise and return words only for seed
        words = [t for t in out if re.match(r"[A-Za-zÀ-ÿ']+$", t)]
        return words[:n_words]

    def generate_seed_text(self, n_words: int = 12, rng: random.Random | None = None) -> str:
        rng = rng or random.Random()
        words = self.generate_words(n_words, rng)
        return " ".join(words)