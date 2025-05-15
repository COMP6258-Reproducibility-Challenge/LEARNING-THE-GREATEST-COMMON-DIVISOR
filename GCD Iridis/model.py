import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Union, List


class MathTokenizer:
    def __init__(self, base: int = 10):
        self.base = base
        self.special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
        self.pad_token, self.sos_token, self.eos_token, self.unk_token = self.special_tokens

        self.digits = [str(i) for i in range(base)]

        # Vocabulary: special tokens + signs + digit symbols
        self.vocab = self.special_tokens + ['+', '-'] + self.digits
        self.token2id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def _int_to_base(self, n: int) -> List[str]:
        if n == 0:
            return [self.digits[0]]
        digits: List[str] = []
        while n > 0:
            digits.append(self.digits[n % self.base])
            n //= self.base
        return list(reversed(digits))

    def encode(self, sequence: Union[str, List[str]]) -> List[int]:
        if isinstance(sequence, str):
            tokens = [self.sos_token] + list(sequence) + [self.eos_token]
        else:
            tokens = [self.sos_token] + sequence + [self.eos_token]
        return [self.token2id.get(tok, self.token2id[self.unk_token]) for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        tokens = [self.id2token.get(i, self.unk_token) for i in ids]
        return [tok for tok in tokens if tok not in (self.sos_token, self.eos_token, self.pad_token)]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class GCDTransformer(nn.Module):
    def __init__(self, tokenizer, d_model=128, nhead=8, num_layers=3, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, self.vocab_size)
        self.pad_id = tokenizer.token2id[tokenizer.pad_token]

    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_key_padding_mask = (src == self.pad_id)
        tgt_key_padding_mask = (tgt == self.pad_id)
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.fc_out(output)

    def predict(self, a: int, b: int, max_length: int = 20) -> int | None:
        digits_a = self.tokenizer._int_to_base(abs(a))
        digits_b = self.tokenizer._int_to_base(abs(b))
        src_tokens = ['+'] + digits_a + ['+'] + digits_b
        src_ids = self.tokenizer.encode(src_tokens)
        src = torch.tensor(src_ids, device=next(self.parameters()).device).unsqueeze(0)
        sos_id = self.tokenizer.token2id[self.tokenizer.sos_token]
        eos_id = self.tokenizer.token2id[self.tokenizer.eos_token]
        tgt_ids = [sos_id]
        for _ in range(max_length):
            tgt = torch.tensor(tgt_ids, device=src.device).unsqueeze(0)
            with torch.no_grad():
                logits = self(src, tgt)
            next_id = logits.argmax(-1)[0, -1].item()
            tgt_ids.append(next_id)
            if next_id == eos_id:
                break
        pred_tokens = self.tokenizer.decode(tgt_ids[1:])
        try:
            value = 0
            for tok in pred_tokens:
                if tok == '+':
                    continue
                digit = self.tokenizer.digits.index(tok)
                value = value * self.tokenizer.base + digit
            return value
        except Exception:
            return None