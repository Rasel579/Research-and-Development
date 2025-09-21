import regex as regex
import json

from tqdm import tqdm

from bpe.base import Tokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = regex.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size > 256
        num_merges = vocab_size - 256
        text_chunks = regex.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in tqdm(range(num_merges), total=num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]]+vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} {vocab[idx]} had {stats[pair]} occurances")
            self.merges = merges
            self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        pair_bytes = []
        for idx in ids:
            if idx in self.vocab:
                pair_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                pair_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"unknown special token {idx}")

        text_bytes = b"".join(pair_bytes)
        return text_bytes.decode("utf-8", errors="replace")

    def _encode_chunks(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats,key= lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        text_chunks = regex.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunks(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special_tokens="none_raise"):
        special = None
        if allowed_special_tokens == "all":
            special = self.special_tokens
        elif allowed_special_tokens == "none":
            special = {}
        elif allowed_special_tokens == "none_raise":
            special = {}
            assert all(token not in text for token in special)
        elif isinstance(allowed_special_tokens, set):
            special = {k: v for k,v in self.special_tokens.items()
                              if k in allowed_special_tokens}
        else:
            raise ValueError(f"unknown special token {allowed_special_tokens}")

        if not special:
            return self.encode_ordinary(text)
        special_patterns = "(" +"|".join(regex.escape(k) for k in special) + ")"
        special_chunks = regex.split(special_patterns, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

if __name__ == "__main__":
    tokenizer = RegexTokenizer()
    tokenizer.load(model_file='../output/tokenizer/tokenzier_v2.model')
    with open('../output/finetune_text_corpus.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    tokenized_data = []
    for item in data:
        tokenized_item = tokenizer.encode(item, "all")
        tokenized_data.append(tokenized_item)


