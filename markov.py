from __future__ import annotations
import re
import json
import gzip
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random
import os

_word_re = re.compile(r"[A-Za-zÀ-ÿ']+|[.?!]")

def tokenize(text: str) -> List[str]:
    # Simple tokenization: words and sentence punctuation
    return _word_re.findall(text)

class MarkovChain:
    """
    Markov chain with weighted transitions and (de)serialization support.

    transitions: Dict[state_tuple, Dict[next_token, count]]
    starts: List[state_tuple] (possible starting states)
    """
    def __init__(self, order: int = 2):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.transitions: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
            self.transitions[state][nxt] += 1

    def _next_weighted(self, state: Tuple[str, ...], rng: random.Random) -> Optional[str]:
        dist = self.transitions.get(state)
        if not dist:
            return None
        total = sum(dist.values())
        r = rng.randrange(total)
        cum = 0
        for tok, count in dist.items():
            cum += count
            if r < cum:
                return tok
        # Should not happen
        return None

    def generate_words(self, n_words: int, rng: random.Random) -> List[str]:
        if not self.starts:
            return []
        state = rng.choice(self.starts)
        out: List[str] = list(state)
        # Count alpha tokens (ignore punctuation tokens)
        def alpha_count(seq: List[str]) -> int:
            return sum(1 for t in seq if re.match(r"[A-Za-zÀ-ÿ']+$", t))

        while alpha_count(out) < n_words:
            nxt = self._next_weighted(tuple(out[-self.order:]), rng)
            if nxt is None:
                # restart from a random start
                state = rng.choice(self.starts)
                out.extend(state)
                continue
            out.append(nxt)
        # Strip trailing punctuation noise and return words-only seed
        words = [t for t in out if re.match(r"[A-Za-zÀ-ÿ']+$", t)]
        return words[:n_words]

    def generate_seed_text(self, n_words: int = 12, rng: Optional[random.Random] = None) -> str:
        rng = rng or random.Random()
        words = self.generate_words(n_words, rng)
        return " ".join(words)

    def to_dict(self) -> dict:
        # Use '|||' joined strings for tuple keys to make JSON compact
        transitions_serialized = {
            "|||".join(state): next_map for state, next_map in self.transitions.items()
        }
        starts_serialized = ["|||".join(state) for state in self.starts]
        return {
            "version": 1,
            "order": self.order,
            "starts": starts_serialized,
            "transitions": transitions_serialized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarkovChain":
        order = int(data.get("order", 2))
        mc = cls(order=order)
        starts_serialized = data.get("starts", [])
        transitions_serialized: Dict[str, Dict[str, int]] = data.get("transitions", {})
        mc.starts = [tuple(s.split("|||")) for s in starts_serialized]
        mc.transitions = defaultdict(lambda: defaultdict(int))
        for state_str, next_map in transitions_serialized.items():
            state = tuple(state_str.split("|||"))
            mc.transitions[state] = defaultdict(int, next_map)
        return mc

    def save(self, path: str):
        """
        Save as JSON or gzipped JSON, based on file extension.
        - .json -> plain JSON
        - .json.gz or .gz -> gzipped JSON
        """
        data = self.to_dict()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MarkovChain":
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return cls.from_dict(data)