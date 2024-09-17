"""
Original :
"Natural Language Processing with Transformers Book"
Tunstall et. al

From:
"Attention as Soft Dictionary Lookup" Yuan Meng
https://www.yuan-meng.com/posts/attention_as_dict/

"""

from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
 
 

# load tokenizer from model checkpoint
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# load config associated with given model
config = AutoConfig.from_pretrained(model_ckpt)

# input text
text = "time flies like an arrow"

# tokenize input text
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

# nn.Embedding is a lookup table to find embeddings of each input_id
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

# look up embeddings by id
input_embs = token_emb(inputs.input_ids)


query,key,value = input_embs,input_embs, input_embs

def scaled_dot_product_attention(query, key,value):
    # hidden dim
    dim_k = key.size(-1)
    # attention weights
    weights = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
    # attenion scores
    scores = F.softmax(weights, dim=-1)
    # compute context vector
    return torch.bmm(scores,value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self,hidden_state):
       attn_outputs = scaled_dot_product_attention(
          self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
       )
       return attn_outputs
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
      super().__init__()
      # bert base uncaesed is embedding size 768
      embed_dim = config.hidden_size
      num_heads = config.num_attention_heads
      # if we have 12 heads, each head get 768 // 12 = 54 hidden_dim
      head_dim = embed_dim // num_heads
      # create list of attention heads
      self.heads = nn.ModuleList(
         [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
      )
      self.output_linear = nn.Linear(embed_dim, embed_dim)
      
    def forward(self, hidden_state):
       # concat output from each head on the last dim
       x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
       # pass through final linear layer
       x = self.output_linear(x)
       return x


# use model config from the beginning
multihead_attn = MultiHeadAttention(config)

# attention outputs concatenated from 12 heads
attn_output = multihead_attn(input_embs)
      


class FeedForward(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
      self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
      self.gelu = nn.GELU()
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, x):
      x = self.linear_1(x)
      x = self.gelu(x)
      x = self.linear_2(x)
      x = self.dropout(x)
      return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
      self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
      self.attention = MultiHeadAttention(config)
      self.feed_forward = FeedForward(config)

    def forward(self, x):
      # apply layer norm on input
      hidden_state = self.layer_norm_1(x)
      # apply attention with skip connection
      x =  x + self.attention(hidden_state)
      # apply feedforward with skip connection
      x = x + self.feed_forward(self.layer_norm_2(x))
      return x
    

class Embeddings(nn.Module):
    def __init__(self, config):
       super().__init__()
       # look up token and position embeddings
       self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
       self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
       # define layernorm and dropout layers
       self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
       self.drop_out = nn.Dropout()
    
    def forward(self, input_ids):
        # length of the input sequence
        seq_length = input_ids.size(1)
        # position id: [0 to seq_length - 1]
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # look up embeddings by id
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # add up token and position embeddings
        embeddings = token_embeddings + position_embeddings
        # pass through layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.drop_out(embeddings)
        return embeddings

class TransformerEncoder(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.embeddings = Embeddings(config)
      self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
   def forward(self, x):
    x = self.embeddings(x)
    for layer in self.layers:
       x = layer(x)
    return x
   

class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,x):
      # select [CLS] token
      x = self.encoder(x)[:,0,:]
      # apply dropout
      x = self.dropout(x)
      # pass through classification layer
      x = self.classifier(x)
      return x
   
config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
encoder_classifier(inputs.input_ids).size()