import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    return


class AttentionHead(nn.Module):
    """
    implementation of a single attention head
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super(AttentionHead, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.qry = nn.Linear(n_embd, head_size, bias=False)
        self.val = nn.Linear(n_embd, head_size, bias=False)
        self.tril = nn.Parameter(
            torch.tril(torch.ones(block_size, block_size)), requires_grad=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.qry(x)
        weight = q @ k.transpose(-2, -1) * (k.shape[-1]) ** (-0.5)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, -torch.inf)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        v = self.val(x)
        return weight @ v


class MultiAttentionHead(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):
        super(MultiAttentionHead, self).__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size, n_embd, block_size, dropout)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class Block(nn.Module):
    """
    a single transformer block
    """

    def __init__(self, n_embd, n_head, block_size, dropout, ff_hidden_dim):
        super(Block, self).__init__()
        assert n_embd % n_head == 0, "n_embd must be divisble by n_head"

        self.head_size = n_embd // n_head
        self.attention = MultiAttentionHead(
            n_head, self.head_size, n_embd, block_size, dropout
        )
        self.linear = FeedForward(n_embd, ff_hidden_dim, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # x + ... adds residual encoding
        x = x + self.linear(self.ln2(x))  # x + ... adds residual encoding
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        block_size,
        n_head,
        dropout,
        ff_hidden_dim,
        n_layers,
        device,
        lr: float = 1e-4,
    ):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, block_size, dropout, ff_hidden_dim)
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(_init_weights)

        self.optim = AdamW(self.parameters(), lr=lr)

    def forward(self, y):
        _, T = y.shape
        tok_emb = self.tok_embeddings(y)
        pos_emb = self.pos_embeddings(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.lm_head(self.ln(x))
        return x

    def sample(self, x):
        x = self.forward(x)
        # focus on last timestep
        return torch.multinomial(F.softmax(x[:, -1, :], dim=-1), num_samples=1)

    def update_parameters(self, x, y, update_weights=True):
        if update_weights:
            self.optim.zero_grad()
            pred = self.forward(x)
            B, T, C = pred.shape
            pred = pred.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            self.optim.step()
        else:
            pred = self.forward(x)
            B, T, C = pred.shape
            pred = pred.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(pred, y)
        return loss.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 512
    n_embd = 384
    n_head = 6
    n_layers = 6
    dropout = 0.2
    block_size = 256
    ff_hidden_dim = 4 * n_embd
    transformer = TransformerDecoder(
        vocab_size, n_embd, block_size, n_head, dropout, ff_hidden_dim, n_layers, device
    ).to(device)

    print(
        f"{sum([p.numel() for p in transformer.parameters()]) / 1e6:2.2f}M parameters"
    )
