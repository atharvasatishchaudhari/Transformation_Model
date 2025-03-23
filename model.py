import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, 
                 d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max(src_seq_len, tgt_seq_len))
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, 
                                          dim_feedforward, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        # src shape: (batch, src_seq_len)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)
        if src_mask is not None:
            src_key_padding_mask = ~(src_mask.squeeze(1).squeeze(1).bool())
        else:
            src_key_padding_mask = None
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # tgt shape: (batch, tgt_seq_len)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # (tgt_seq_len, batch, d_model)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.squeeze(0)
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return output.transpose(0, 1)

    def project(self, dec_output):
        # dec_output shape: (batch, tgt_seq_len, d_model)
        logits = self.generator(dec_output)
        return logits

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512):
    model = TransformerModel(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=d_model)
    return model
