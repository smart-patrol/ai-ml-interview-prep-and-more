import torch
import torch.nn.functional as F

# input sequence / sentence:
#  "Can you help me to translate this sentence"

sentence = torch.tensor(
    [0, # can
     7, # you     
     1, # help
     2, # me
     5, # to
     6, # translate
     4, # this
     3] # sentence
)

print(sentence)


torch.manual_seed(123)
embed = torch.nn.Embedding(10,16)
embedded_setence = embed(sentence).detach()
print(embedded_setence.shape)

# compute attention weights
omega = embedded_setence.matmul(embedded_setence.T)
attention_weights = F.softmax(omega, dim=1)
print(attention_weights.shape)
print(attention_weights.sum(dim=1))

# compute context vectors
context_vectors = torch.matmul(
    attention_weights, embedded_setence
)
