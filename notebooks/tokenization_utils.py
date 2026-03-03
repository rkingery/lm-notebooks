# tokenization_utils.py — BPE training and encoding helpers for 1a-tokenization.ipynb.
# Worker functions (pretokenize, encode) live here so they can be pickled by spawn-based
# multiprocessing. train_bpe is defined in the notebook and calls into these helpers.

import os
import re
import json
import csv
import heapq
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import regex
from tqdm import tqdm

gpt2_pretokenizer_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@lru_cache(maxsize=None)
def _heap_sort_key(b):
    return tuple(256 - x for x in b) + (257,)

def heap_push(heap, pair, count, id_to_bytes):
    a, b = pair
    heapq.heappush(heap, (-count, _heap_sort_key(id_to_bytes[a]), _heap_sort_key(id_to_bytes[b]), a, b))

def heap_pop(heap, pair_counts):
    while heap:
        neg_count, _, _, a, b = heapq.heappop(heap)
        pair = (a, b)
        if pair_counts.get(pair, 0) == -neg_count:
            return pair
    return None

def _find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b'':
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def _pretokenize_chunk(args):
    input_path, start, end, special_pattern, pattern = args
    chunk_freqs = {}
    word_cache = {}
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    segments = re.split(special_pattern, chunk) if special_pattern else [chunk]
    for segment in segments:
        for match in regex.finditer(pattern, segment):
            word = match.group()
            if word not in word_cache:
                word_cache[word] = tuple(word.encode('utf-8'))
            tok = word_cache[word]
            chunk_freqs[tok] = chunk_freqs.get(tok, 0) + 1
    return chunk_freqs

def pretokenize(input_path, special_tokens, pattern):
    special_pattern = '|'.join(re.escape(t) for t in special_tokens) if special_tokens else None
    split_token = special_tokens[0].encode('utf-8') if special_tokens else None
    num_procs = multiprocessing.cpu_count()
    with open(input_path, 'rb') as file:
        boundaries = _find_chunk_boundaries(file, num_procs * 8, split_token) if split_token else [0, os.path.getsize(input_path)]
    chunk_args = [(input_path, start, end, special_pattern, pattern) for start, end in zip(boundaries[:-1], boundaries[1:])]
    pretoken_freqs = {}
    if len(chunk_args) <= 1:
        return _pretokenize_chunk(chunk_args[0]) if chunk_args else pretoken_freqs
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        for future in as_completed(executor.submit(_pretokenize_chunk, arg) for arg in chunk_args):
            for word, count in future.result().items():
                pretoken_freqs[word] = pretoken_freqs.get(word, 0) + count
    return pretoken_freqs

def get_pair_counts_and_index(pretoken_freqs):
    pretoken_seqs = {}
    pretoken_freqs_by_id = {}
    pair_counts = defaultdict(int)
    pair_index = {}
    for pid, (tok_tuple, freq) in enumerate(pretoken_freqs.items()):
        pretoken_seqs[pid] = tok_tuple
        pretoken_freqs_by_id[pid] = freq
        for pair in zip(tok_tuple, tok_tuple[1:]):
            pair_counts[pair] += freq
            pair_index.setdefault(pair, set()).add(pid)
    return pretoken_seqs, pretoken_freqs_by_id, pair_counts, pair_index

def merge_pair(pretoken_seqs, pretoken_freqs_by_id, pair_counts, pair, new_id, pair_index):
    a, b = pair
    modified = set()
    for pid in list(pair_index.get(pair, set())):
        seq = pretoken_seqs[pid]
        freq = pretoken_freqs_by_id[pid]
        old_pairs = set(zip(seq, seq[1:]))
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                if new_seq:
                    left = new_seq[-1]
                    pair_counts[(left, a)] -= freq
                    pair_counts[(left, new_id)] += freq
                    modified.add((left, a)); modified.add((left, new_id))
                if i + 2 < len(seq):
                    right = seq[i + 2]
                    pair_counts[(b, right)] -= freq
                    pair_counts[(new_id, right)] += freq
                    modified.add((b, right)); modified.add((new_id, right))
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_seq_t = tuple(new_seq)
        new_pairs = set(zip(new_seq_t, new_seq_t[1:]))
        for p in old_pairs - new_pairs:
            s = pair_index.get(p)
            if s: s.discard(pid)
        for p in new_pairs - old_pairs:
            pair_index.setdefault(p, set()).add(pid)
        pretoken_seqs[pid] = new_seq_t
    pair_counts[pair] = 0
    modified.add(pair)
    return modified

# ── encoding ──────────────────────────────────────────────────────────────────

_reverse_vocab = None
_merge_ranks = None
_special_set = None
_split_pat = None
_pattern = None

def _bpe_apply(pretoken, merge_ranks):
    tokens = [bytes([b]) for b in pretoken.encode('utf-8')]
    while len(tokens) >= 2:
        best_rank, best_idx = float('inf'), -1
        for i in range(len(tokens) - 1):
            rank = merge_ranks.get((tokens[i], tokens[i + 1]), float('inf'))
            if rank < best_rank:
                best_rank, best_idx = rank, i
        if best_idx == -1:
            break
        a, b = tokens[best_idx], tokens[best_idx + 1]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

def _init_encode_worker(vocab_path, merges_path, special_tokens, pattern):
    global _reverse_vocab, _merge_ranks, _special_set, _split_pat, _pattern
    with open(vocab_path) as f:
        vocab = {int(id): bytes(b) for id, b in json.load(f).items()}
    with open(merges_path, newline='') as f:
        _merge_ranks = {(bytes.fromhex(r[0]), bytes.fromhex(r[1])): i for i, r in enumerate(csv.reader(f))}
    _reverse_vocab = {v: k for k, v in vocab.items()}
    _special_set = set(special_tokens)
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    _split_pat = '(' + '|'.join(re.escape(t) for t in sorted_tokens) + ')' if special_tokens else None
    _pattern = pattern

def _encode_chunk(args):
    input_path, start, end = args
    with open(input_path, 'rb') as file:
        file.seek(start)
        text = file.read(end - start).decode('utf-8', errors='replace')
    chunks = re.split(_split_pat, text) if _split_pat else [text]
    ids = []
    for chunk in chunks:
        if not chunk:
            continue
        if chunk in _special_set:
            ids.append(_reverse_vocab[chunk.encode('utf-8')])
            continue
        for match in regex.finditer(_pattern, chunk):
            ids.extend(_reverse_vocab[token] for token in _bpe_apply(match.group(), _merge_ranks))
    return ids

def tokenize_dataset_fast(input_path, output_path, vocab_path, merges_path, special_tokens=None,
                           pattern=gpt2_pretokenizer_pattern, num_procs=None, chunks_per_proc=32):
    if special_tokens is None:
        special_tokens = ['<|endoftext|>']
    if num_procs is None:
        num_procs = os.cpu_count()
    with open(input_path, 'rb') as file:
        boundaries = _find_chunk_boundaries(file, num_procs * chunks_per_proc, special_tokens[0].encode())
    args = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    chunk_sizes = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    with open(output_path, 'wb') as out:
        with ProcessPoolExecutor(
            max_workers=num_procs, initializer=_init_encode_worker, initargs=(vocab_path, merges_path, special_tokens, pattern)
        ) as executor:
            with tqdm(total=os.path.getsize(input_path), unit='B', unit_scale=True) as pbar:
                for chunk_tokens, chunk_size in zip(executor.map(_encode_chunk, args), chunk_sizes):
                    np.array(chunk_tokens, dtype=np.uint16).tofile(out)
                    pbar.update(chunk_size)
