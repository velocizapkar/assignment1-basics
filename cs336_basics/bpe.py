import base64
import json
import os

from collections import Counter
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

from cs336_basics.names import generate_name

NUM_PROCESSES = 4
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
BYTE_LOOKUP = tuple(bytes([i]) for i in range(256))


def find_chunk_boundaries(
    file: BinaryIO,
    num_processes: int,
    special_tokens: list[str],
) -> list[int]:
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // num_processes

    chunk_boundaries = [i * chunk_size for i in range(num_processes + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            for token in special_tokens:
                token_position = mini_chunk.find(token.encode('utf-8'))
                if token_position != -1:
                    chunk_boundaries[bi] = initial_position + token_position
                    break

            if chunk_boundaries[bi] != initial_position:
                break

            initial_position += mini_chunk_size
    
    return sorted(set(chunk_boundaries))


def count_tokens_from_file(args: tuple[str, int, int, list[str], ]) -> Counter[tuple]:
    """Worker reads its own chunk from disk."""
    path, start, end, special_tokens, split_pattern = args
    with open(path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(split_pattern, chunk)
    else:
        parts = [chunk]

    counter = Counter()
    for part in parts:
        for word in PAT.findall(part):
            word_bytes = word.encode('utf-8')
            # Use lookup table instead of bytes([b]) - avoids allocation
            counter[tuple(BYTE_LOOKUP[b] for b in word_bytes)] += 1

    return counter


def pretokenize(
    input_path: str,
    special_tokens: list[str],
    num_processes: int
) -> dict[tuple]:
    
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)

    # Pre-compile split pattern once, not per-worker
    split_pattern = None
    if special_tokens:
        split_pattern = re.compile("|".join(re.escape(token) for token in special_tokens))

    work_items = [
        (input_path, start, end, special_tokens, split_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        counters = pool.map(count_tokens_from_file, work_items)
    
    return dict(sum(counters, Counter()))


class BPETrainer:
    def __init__(self, vocab, word_freqs: dict[tuple]):
        self.vocab = vocab
        self.word_counts = word_freqs

        self.pair_counts = defaultdict(int)
        self.pair_to_words = defaultdict(set)
        
        for word, freq in self.word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                self.pair_counts[pair] += freq
                self.pair_to_words[pair].add(word)
    
    def best_pair(self) -> tuple | None:
        if not self.pair_counts:
            return None
        # tiebreak by lexicographic order for determinism
        return max(self.pair_counts, key=lambda p: (self.pair_counts[p], p))
    
    def merge(self, pair: tuple) -> str:
        merged_token = pair[0] + pair[1]
        affected_words = list(self.pair_to_words.pop(pair, set()))
        del self.pair_counts[pair]
        
        for word in affected_words:
            freq = self.word_counts.pop(word)
            
            # decrement old pairs
            for i in range(len(word) - 1):
                p = (word[i], word[i+1])
                self.pair_counts[p] -= freq
                if self.pair_counts[p] <= 0:
                    del self.pair_counts[p]
                self.pair_to_words[p].discard(word)
            
            # build merged word
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            
            # add new word
            self.word_counts[new_word] = freq
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i+1])
                self.pair_counts[p] += freq
                self.pair_to_words[p].add(new_word)
        
        return merged_token


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # init vocab (special tokens + 256 bytes)
    special_regions = [i.encode('utf-8') for i in special_tokens]
    byte_regions = [bytes([i]) for i in range(256)]
    vocab = {i:token for i, token in enumerate(special_regions + byte_regions)}

    # pre-tokenize
    word_counts = pretokenize(input_path, special_tokens, NUM_PROCESSES)

    # compute merges
    trainer = BPETrainer(vocab, word_counts)
    merges = []
    
    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        pair = trainer.best_pair()
        if pair is None:
            break
        merged = trainer.merge(pair)
        vocab[len(vocab)] = merged
        merges.append((pair[0], pair[1]))
    
    return vocab, merges


def save_vocab(vocab_filepath: str, vocab: dict[int, bytes]):
    with open(vocab_filepath, 'w') as f:
        for idx, token_bytes in sorted(vocab.items()):
            f.write(f'{base64.b64encode(token_bytes).decode()} {idx}\n')


def save_merges(merges_filepath: str, merges: list[tuple[bytes, bytes]]):
    with open(merges_filepath, 'w') as f:
        for b1, b2 in merges:
            f.write(f'{base64.b64encode(b1).decode()} {base64.b64decode(b2).decode()}\n')


def save_special_tokens(special_tokens_filepath: str, special_tokens: list[str]):
    with open(special_tokens_filepath, 'w') as f:
        json.dump(special_tokens, f)


def save_tokenizer(scheme: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
    save_vocab(f'{scheme}/vocab.bpe', vocab)
    save_merges(f'{scheme}/merges.txt', merges)
    save_special_tokens(f'{scheme}/special_tokens.json', special_tokens)


def load_vocab(vocab_filepath: str):
    vocab = {}
    with open(vocab_filepath, 'r') as f:
        for line in f:
            token_b64, idx = line.strip().split()
            vocab[int(idx)] = base64.b64decode(token_b64)

    return vocab


def load_merges(merges_filepath: str):
    merges = []
    with open(merges_filepath, 'r') as f:
        for line in f:
            b1, b2 = line.strip().split()
            merges.append((base64.b64decode(b1), base64.b64decode(b2)))

    return merges


def load_special_tokens(special_tokens_filepath: str):
    special_tokens = []
    with open(special_tokens_filepath, 'r') as f:
        special_tokens = json.load(f)

    return special_tokens


def load_tokenizer(scheme: str):
    vocab = load_vocab(f'{scheme}/vocab.bpe')
    merges = load_merges(f'{scheme}/merges.txt')
    special_tokens = load_special_tokens(f'{scheme}/special_tokens.json')
    
    return vocab, merges, special_tokens


if __name__ == "__main__":
    run_name = generate_name()

    dataset = 'TinyStoriesV2-GPT4-valid.txt'
    vocab_size = 500
    special_tokens = ['<|endoftext|>']

    stem = os.path.splitext(dataset)[0]
    scheme = f'artifacts/bpe/{run_name}__{stem}__{vocab_size}'

    os.makedirs(scheme, exist_ok=True)

    vocab, merges = train_bpe(f'data/{dataset}', vocab_size, special_tokens)
    save_tokenizer(scheme, vocab, merges, special_tokens)
