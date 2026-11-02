import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        # TODO: token_emb, pos_emb, blocks (ModuleList of TransformerEncoderLayer), ln_f, head
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead = num_heads, dim_feedforward = 4*d_model, activation = 'gelu', batch_first = True, norm_first = True, dropout = 0.0) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias = False)



    def forward(self, idx):
        # TODO: embed + pos, run through blocks with causal mask, final LN, project to vocab
        B, T = idx.shape 
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        mask = mask.to(idx.device)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
