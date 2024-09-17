import torch
import torch.nn as nn
from torchtyping import TensorType

class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        
        # Word and positional embeddings
        self.embd = nn.Embedding(vocab_size, model_dim)
        self.pos_embd = nn.Embedding(context_length, model_dim)
        
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[self.TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)]
        )
        
        # Final layer normalization and linear projection
        self.norm = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        # Get token embeddings
        embd = self.embd(context)
        
        # Get positional embeddings
        context_length = context.shape[1]
        positions = torch.arange(context_length, device=context.device)
        embed = embd + self.pos_embd(positions)
        
        # Pass through transformer blocks and final layer norm
        out = self.norm(self.blocks(embed))
        
        # Project to vocabulary size
        logits = self.linear(out)
        
        return logits

    def generate(self, context: TensorType[int], max_new_tokens: int) -> TensorType[int]:
        for _ in range(max_new_tokens):
            logits = self(context)
            logits = logits[:, -1, :]  # Get logits for the last token
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
        return context

    # Transformer Block
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    scores = q @ torch.transpose(k, 1, 2)
                    attention_dim = k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    # Apply mask to prevent attending to future tokens
                    context_length = k.shape[1]
                    lower_triangular = torch.tril(torch.ones(context_length, context_length, device=embedded.device))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=-1)

                    return scores @ v
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                self.att_heads = nn.ModuleList(
                    [self.SingleHeadAttention(model_dim, model_dim // num_heads) for _ in range(num_heads)]
                )

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = [head(embedded) for head in self.att_heads]
                concatenated = torch.cat(head_outputs, dim=2)
                return concatenated
        
        class FeedForwardNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.feed_forward = self.FeedForwardNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # Apply attention with residual connection
            embedded = embedded + self.attention(self.first_norm(embedded))
            # Apply feed-forward network with residual connection
            embedded = embedded + self.feed_forward(self.second_norm(embedded))
            return embedded

# Example usage
vocab_size = 50257
context_length = 1024
model_dim = 768
num_blocks = 12
num_heads = 12

model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)

# Example input
input_ids = torch.randint(0, vocab_size, (1, context_length))

# Forward pass
logits = model(input_ids)

# Generate new tokens
generated_ids = model.generate(input_ids, max_new_tokens=50)
