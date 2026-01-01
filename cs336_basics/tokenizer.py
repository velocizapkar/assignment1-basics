import regex as re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from cs336_basics.bpe import PAT, BYTE_LOOKUP, load_vocab, load_merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_set = frozenset(self.special_tokens)
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.max_special = max((len(t) for t in self.special_tokens), default=0)
        # Longest-first for correct overlapping match
        if self.special_tokens:
            pat = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self.special_pat = re.compile(f"({pat})")
        else:
            self.special_pat = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        return cls(load_vocab(vocab_filepath), load_merges(merges_filepath), special_tokens)

    def encode(self, text: str) -> list[int]:
        parts = self.special_pat.split(text) if self.special_pat else [text]
        result = []
        for part in parts:
            if not part:
                continue
            if part in self.special_set:
                result.append(self.reverse_vocab[part.encode('utf-8')])
                continue
            # Pretokenize into byte sequences
            pretokens = [[BYTE_LOOKUP[b] for b in w.encode('utf-8')] for w in PAT.findall(part)]
            # Index: pair -> set of pretoken indices containing it
            pair_idx = defaultdict(set)
            for i, pt in enumerate(pretokens):
                for j in range(len(pt) - 1):
                    pair_idx[(pt[j], pt[j + 1])].add(i)
            # Apply merges in priority order
            for pair in self.merges:
                if pair not in pair_idx:
                    continue
                merged = pair[0] + pair[1]
                for i in list(pair_idx.pop(pair)):
                    pt = pretokens[i]
                    # Remove old pairs from index
                    for j in range(len(pt) - 1):
                        pair_idx[(pt[j], pt[j + 1])].discard(i)
                    # Apply merge
                    new_pt, j = [], 0
                    while j < len(pt):
                        if j < len(pt) - 1 and (pt[j], pt[j + 1]) == pair:
                            new_pt.append(merged)
                            j += 2
                        else:
                            new_pt.append(pt[j])
                            j += 1
                    pretokens[i] = new_pt
                    # Re-index new pairs
                    for j in range(len(new_pt) - 1):
                        pair_idx[(new_pt[j], new_pt[j + 1])].add(i)
            # Convert tokens to IDs
            for pt in pretokens:
                result.extend(self.reverse_vocab[t] for t in pt)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buf, margin = "", max(self.max_special - 1, 0)
        for text in iterable:
            buf += text
            if len(buf) > margin:
                # Split BEFORE whitespace to preserve GPT-2 pretoken boundaries (" word")
                split = max(buf.rfind(c, 0, len(buf) - margin) for c in " \n\t\r")
                if split > 0:
                    yield from self.encode(buf[:split])
                    buf = buf[split:]
        if buf:
            yield from self.encode(buf)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')
