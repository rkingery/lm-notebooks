import json
import time
from functools import lru_cache

import tiktoken

from .common import FIXTURES_PATH, Snapshot

@lru_cache
def gpt2_bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))

def test_train_bpe_speed(run_train_bpe):
    input_path = FIXTURES_PATH / 'corpus.en'
    start_time = time.time()
    _, _ = run_train_bpe(input_path=input_path, vocab_size=500, special_tokens=['<|endoftext|>'])
    end_time = time.time()
    assert end_time - start_time < 1.5, f"BPE training took {end_time - start_time:.2f}s (limit 1.5s)"
    print('PASSED: test_train_bpe_speed')

def test_train_bpe(run_train_bpe):
    input_path = FIXTURES_PATH / 'corpus.en'
    vocab, merges = run_train_bpe(input_path=input_path, vocab_size=500, special_tokens=['<|endoftext|>'])
    reference_vocab_path = FIXTURES_PATH / 'train-bpe-reference-vocab.json'
    reference_merges_path = FIXTURES_PATH / 'train-bpe-reference-merges.txt'
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding='utf-8') as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(' ')) for line in f]
        reference_merges = [
            (bytes([gpt2_byte_decoder[t] for t in m1]), bytes([gpt2_byte_decoder[t] for t in m2]))
            for m1, m2 in gpt2_reference_merges
        ]
    assert merges == reference_merges
    with open(reference_vocab_path, encoding='utf-8') as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {idx: bytes([gpt2_byte_decoder[t] for t in item]) for item, idx in gpt2_reference_vocab.items()}
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())
    print('PASSED: test_train_bpe')

def test_train_bpe_special_tokens(run_train_bpe):
    snap = Snapshot('test_train_bpe_special_tokens')
    input_path = FIXTURES_PATH / 'tinystories_sample_5M.txt'
    vocab, merges = run_train_bpe(input_path=input_path, vocab_size=1000, special_tokens=['<|endoftext|>'])
    vocabs_without_specials = [word for word in vocab.values() if word != b'<|endoftext|>']
    for word_bytes in vocabs_without_specials:
        assert b'<|' not in word_bytes
    snap.assert_match({'vocab_keys': set(vocab.keys()), 'vocab_values': set(vocab.values()), 'merges': merges})
    print('PASSED: test_train_bpe_special_tokens')

VOCAB_PATH = FIXTURES_PATH / 'gpt2_vocab.json'
MERGES_PATH = FIXTURES_PATH / 'gpt2_merges.txt'

def _make_tokenizer(get_tokenizer_fn, special_tokens=None):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(VOCAB_PATH) as f:
        gpt2_vocab = json.load(f)
    gpt2_bpe_merges = []
    with open(MERGES_PATH) as f:
        for line in f:
            cleaned = line.rstrip()
            if cleaned and len(cleaned.split(' ')) == 2:
                gpt2_bpe_merges.append(tuple(cleaned.split(' ')))
    vocab = {idx: bytes([gpt2_byte_decoder[tok] for tok in item]) for item, idx in gpt2_vocab.items()}
    if special_tokens:
        for st in special_tokens:
            bst = st.encode('utf-8')
            if bst not in set(vocab.values()):
                vocab[len(vocab)] = bst
    merges = [(bytes([gpt2_byte_decoder[t] for t in m1]), bytes([gpt2_byte_decoder[t] for t in m2])) for m1, m2 in gpt2_bpe_merges]
    return get_tokenizer_fn(vocab, merges, special_tokens)

def test_roundtrip_empty(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    s = ''
    assert tokenizer.decode(tokenizer.encode(s)) == s
    print('PASSED: test_roundtrip_empty')

def test_roundtrip_single_character(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    s = 's'
    assert tokenizer.decode(tokenizer.encode(s)) == s
    print('PASSED: test_roundtrip_single_character')

def test_roundtrip_single_unicode_character(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    s = '\U0001f643'
    assert tokenizer.decode(tokenizer.encode(s)) == s
    print('PASSED: test_roundtrip_single_unicode_character')

def test_roundtrip_ascii_string(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    s = 'Hello, how are you?'
    assert tokenizer.decode(tokenizer.encode(s)) == s
    print('PASSED: test_roundtrip_ascii_string')

def test_roundtrip_unicode_string(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    s = 'H\u00e9ll\u00f2 h\u00f4w are \u00fc? \U0001f643'
    assert tokenizer.decode(tokenizer.encode(s)) == s
    print('PASSED: test_roundtrip_unicode_string')

def test_roundtrip_unicode_string_with_special_tokens(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>'])
    s = 'H\u00e9ll\u00f2 h\u00f4w <|endoftext|><|endoftext|> are \u00fc? \U0001f643<|endoftext|>'
    ids = tokenizer.encode(s)
    tokenized = [tokenizer.decode([x]) for x in ids]
    assert tokenized.count('<|endoftext|>') == 3
    assert tokenizer.decode(ids) == s
    print('PASSED: test_roundtrip_unicode_string_with_special_tokens')

def test_address_roundtrip(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    with open(FIXTURES_PATH / 'address.txt') as f:
        text = f.read()
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_address_roundtrip')

def test_german_roundtrip(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    with open(FIXTURES_PATH / 'german.txt') as f:
        text = f.read()
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_german_roundtrip')

def test_tinystories_sample_roundtrip(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    with open(FIXTURES_PATH / 'tinystories_sample.txt') as f:
        text = f.read()
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_tinystories_sample_roundtrip')

def test_empty_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer)
    s = ''
    ref_ids = ref.encode(s)
    ids = tokenizer.encode(s)
    assert ids == ref_ids
    assert [tokenizer.decode([x]) for x in ids] == []
    assert tokenizer.decode(ids) == s
    print('PASSED: test_empty_matches_tiktoken')

def test_single_character_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer)
    s = 's'
    ref_ids = ref.encode(s)
    ids = tokenizer.encode(s)
    assert ids == ref_ids
    assert [tokenizer.decode([x]) for x in ids] == ['s']
    assert tokenizer.decode(ids) == s
    print('PASSED: test_single_character_matches_tiktoken')

def test_single_unicode_character_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer)
    s = '\U0001f643'
    assert tokenizer.encode(s) == ref.encode(s)
    assert tokenizer.decode(tokenizer.encode(s)) == s
    print('PASSED: test_single_unicode_character_matches_tiktoken')

def test_ascii_string_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer)
    s = 'Hello, how are you?'
    ids = tokenizer.encode(s)
    assert [tokenizer.decode([x]) for x in ids] == ['Hello', ',', ' how', ' are', ' you', '?']
    assert tokenizer.decode(ids) == s
    assert ref.decode(ref.encode(s)) == s
    print('PASSED: test_ascii_string_matches_tiktoken')

def test_unicode_string_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>'])
    s = 'H\u00e9ll\u00f2 h\u00f4w are \u00fc? \U0001f643'
    assert tokenizer.encode(s) == ref.encode(s)
    assert tokenizer.decode(tokenizer.encode(s)) == s
    print('PASSED: test_unicode_string_matches_tiktoken')

def test_unicode_string_with_special_tokens_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>'])
    s = 'H\u00e9ll\u00f2 h\u00f4w <|endoftext|><|endoftext|> are \u00fc? \U0001f643<|endoftext|>'
    ref_ids = ref.encode(s, allowed_special={'<|endoftext|>'})
    ids = tokenizer.encode(s)
    assert ids == ref_ids
    assert tokenizer.decode(ids) == s
    print('PASSED: test_unicode_string_with_special_tokens_matches_tiktoken')

def test_overlapping_special_tokens(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>', '<|endoftext|><|endoftext|>'])
    s = 'Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>'
    ids = tokenizer.encode(s)
    tokenized = [tokenizer.decode([x]) for x in ids]
    assert tokenized.count('<|endoftext|>') == 1
    assert tokenized.count('<|endoftext|><|endoftext|>') == 1
    assert tokenizer.decode(ids) == s
    print('PASSED: test_overlapping_special_tokens')

def test_address_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer)
    with open(FIXTURES_PATH / 'address.txt') as f:
        text = f.read()
    assert tokenizer.encode(text) == ref.encode(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_address_matches_tiktoken')

def test_german_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer)
    with open(FIXTURES_PATH / 'german.txt') as f:
        text = f.read()
    assert tokenizer.encode(text) == ref.encode(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_german_matches_tiktoken')

def test_tinystories_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>'])
    with open(FIXTURES_PATH / 'tinystories_sample.txt') as f:
        text = f.read()
    ref_ids = ref.encode(text, allowed_special={'<|endoftext|>'})
    assert tokenizer.encode(text) == ref_ids
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_tinystories_matches_tiktoken')

def test_encode_special_token_trailing_newlines(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>'])
    with open(FIXTURES_PATH / 'special_token_trailing_newlines.txt') as f:
        text = f.read()
    ref_ids = ref.encode(text, allowed_special={'<|endoftext|>'})
    assert tokenizer.encode(text) == ref_ids
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_encode_special_token_trailing_newlines')

def test_encode_special_token_double_newline_non_whitespace(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>'])
    with open(FIXTURES_PATH / 'special_token_double_newlines_non_whitespace.txt') as f:
        text = f.read()
    ref_ids = ref.encode(text, allowed_special={'<|endoftext|>'})
    assert tokenizer.encode(text) == ref_ids
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('PASSED: test_encode_special_token_double_newline_non_whitespace')

def test_encode_iterable_tinystories_sample_roundtrip(get_tokenizer):
    tokenizer = _make_tokenizer(get_tokenizer)
    all_ids = []
    with open(FIXTURES_PATH / 'tinystories_sample.txt') as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(FIXTURES_PATH / 'tinystories_sample.txt') as f:
        text = f.read()
    assert tokenizer.decode(all_ids) == text
    print('PASSED: test_encode_iterable_tinystories_sample_roundtrip')

def test_encode_iterable_tinystories_matches_tiktoken(get_tokenizer):
    ref = tiktoken.get_encoding('gpt2')
    tokenizer = _make_tokenizer(get_tokenizer, special_tokens=['<|endoftext|>'])
    with open(FIXTURES_PATH / 'tinystories_sample.txt') as f:
        text = f.read()
    ref_ids = ref.encode(text, allowed_special={'<|endoftext|>'})
    all_ids = []
    with open(FIXTURES_PATH / 'tinystories_sample.txt') as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    assert all_ids == ref_ids
    assert tokenizer.decode(all_ids) == text
    print('PASSED: test_encode_iterable_tinystories_matches_tiktoken')
