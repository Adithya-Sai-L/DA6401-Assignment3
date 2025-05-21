PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

class Vocab:
    def __init__(self):
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.char2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.freqs = {}

    def add_sentence(self, sentence: str):
        for ch in sentence:
            if ch not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch
                self.freqs[ch] = 1
            else:
                self.freqs[ch] = self.freqs.get(ch, 0) + 1

    def sentence_to_indices(self, sentence: str) -> list[int]:
        unk_idx = self.char2idx[UNK_TOKEN]
        return [self.char2idx.get(ch, unk_idx) for ch in sentence]

    def indices_to_sentence(self, indices: list[int]) -> str:
        eos_idx = self.char2idx[EOS_TOKEN]
        return ''.join(self.idx2char.get(idx, UNK_TOKEN) for idx in indices if idx != eos_idx)

    @property
    def size(self) -> int:
        return len(self.char2idx)
    

import pandas as pd
import torch
from torch.utils.data import Dataset

class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tsv_file: str,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
    ):
        # Read data efficiently with only needed columns
        self.data = pd.read_csv(
            tsv_file,
            sep='\t',
            header=None,
            usecols=[0, 1],
            names=['dev', 'lat'],
            dtype=str,
            encoding='utf-8',
            na_values=[],
            keep_default_na=False
        )
        
        # Swap columns and convert to string in one operation
        self.data = self.data.rename(columns={'lat': 'src', 'dev': 'tgt'})
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Pre-compute indices for SOS and EOS tokens
        self.sos_idx = self.tgt_vocab.char2idx[SOS_TOKEN]
        self.eos_idx = self.tgt_vocab.char2idx[EOS_TOKEN]

        # Build vocabularies in one pass
        for _, row in self.data.iterrows():
            src_vocab.add_sentence(row['src'])
            tgt_vocab.add_sentence(row['tgt'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Convert to indices
        src_idx = self.src_vocab.sentence_to_indices(row['src'])
        tgt_idx = [self.sos_idx] + self.tgt_vocab.sentence_to_indices(row['tgt']) + [self.eos_idx]
        
        # Convert to tensors
        return (
            torch.tensor(src_idx, dtype=torch.long),
            torch.tensor(tgt_idx, dtype=torch.long)
        )
