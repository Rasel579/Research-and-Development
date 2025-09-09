from typing import Optional, Tuple
import torch
import torch.nn as nn
from  torch.nn import functional as F

def print_model_structure(model: torch.nn.Module, indent: str = '') -> None:
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"{indent}|--{name}: {child.__class__.__name__}: ({params:,} parameters)")
        print_model_structure(child, indent + '|  ')

class Head(nn.Module):
    def __init__(self, head_size: int, n_embeddings: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embeddings: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embeddings, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size*num_heads, n_embeddings, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embedding: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4*n_embedding),
            nn.ReLU(),
            nn.Linear(4*n_embedding, n_embedding),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embedding: int, n_head: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embedding // n_head
        self.attention = MultiHeadAttention(n_head, head_size, n_embedding, dropout)
        self.feed_forward = FeedForward(n_embedding, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embedding)
        self.layer_norm2 = nn.LayerNorm(n_embedding)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embeddings: int, block_size: int, n_head: int, n_layers: int, device: str) -> None:
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.positional_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.blocks = nn.Sequential(*[Block(n_embeddings, n_head= n_head, dropout=dropout) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(n_embeddings)
        self.final_linear_layer = nn.Linear(n_embeddings, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape
        token_embeddings = self.token_embedding_table(input_tokens)
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device=self.device))
        x = token_embeddings + positional_embeddings
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.final_linear_layer(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens

if __name__ == "__main__":
    block_size = 256
    n_embeddings = 384
    n_head = 6
    n_layers = 6
    dropout = 0.2
    vocab_size = 1034
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTLanguageModel(vocab_size=vocab_size, n_embeddings=n_embeddings, block_size=block_size, n_head=n_head, n_layers=n_layers, device=device)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    batch_size = 1
    seq_length = 6
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    x = x.to(device)
    logits, loss = model(x)
    print(logits.shape, loss)
    print_model_structure(model)
