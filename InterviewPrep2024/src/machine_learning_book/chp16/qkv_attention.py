# Importing necessary libraries
import torch
import torch.nn.functional as F

# Defining the sentence as a tensor of word indices
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

# Setting the seed for reproducibility
torch.manual_seed(123)

# Defining an embedding layer with 10 words and 16 dimensions
embed = torch.nn.Embedding(10,16)

# Getting the embedded representation of the sentence
embedded_sentence = embed(sentence).detach()

# Getting the dimension of the embedded sentence
d = embedded_sentence.shape[1]

# Initializing the matrices for query, key, and value transformations
U_query = torch.rand(d,d)
U_key = torch.rand(d,d)
U_value = torch.rand(d,d)

# Getting the query, key, and value vectors for the second word in the sentence
x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

# Calculating the keys for all words in the sentence
keys = U_key.matmul(embedded_sentence.T).T

# Checking if the key for the second word is correct
print(torch.allclose(key_2, keys[1]))

# Calculating the values for all words in the sentence
values = U_value.matmul(embedded_sentence.T).T

# Checking if the value for the second word is correct
print(torch.allclose(value_2, values[1]))

# Calculating the attention weight for the second word with respect to the third word
omega23 = query_2.dot(keys[2])

# Calculating the attention weights with respect to all words in the sentence
omega2 = query_2.matmul(keys.T)
attention_weights = F.softmax(omega2 / d**0.05, dim=0) # scaled to ensure in same range

# Calculating the context vector for the second word
context_vector = attention_weights.matmul(values)
print(context_vector)

# Encoding context embeddings via multi-head attention
torch.manual_seed(123)

# Getting the dimension of the embedded sentence
d = embedded_sentence.shape[1]

#########################################################################
# Encoding context embeddings via multi-head attention


# Initializing the matrices for query, key, and value transformations for one head
one_U_query = torch.rand(d,d)

# Defining the number of heads
head = 8

# Initializing the matrices for query, key, and value transformations for multiple heads
multihead_U_query = torch.rand(head,d,d)
multihead_U_key =  torch.rand(head,d,d)
multihead_U_value = torch.rand(head,d,d)

# Getting the query, key, and value vectors for the second word in the sentence for one head
multihead_query_2 = multihead_U_query.matmul(x_2)
multihead_key_2 = multihead_U_key.matmul(x_2) 
multihead_value_2 = multihead_U_value.matmul(x_2)

# Stacking the embedded sentence for multiple heads
stacked_inputs =  embedded_sentence.T.repeat(8,1,1)

# Calculating the keys for all words in the sentence for multiple heads
# dot product between matrices (head, d,d)  and (head, d,seq_length)
multihead_keys = torch.bmm(multihead_key_2, stacked_inputs) # output : (head, d, seq_len)
multihead_keys = multihead_keys.permute(0,2,1) # rearrange to (head, seq_length, d)

# Calculating the values for all words in the sentence for multiple heads
multihead_values= torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0,2,1)

# Initializing the context vector for multiple heads
multihead_z_2 = torch.rand(8,16)

# Defining a linear layer to combine the context vectors from multiple heads
linear = torch.nn.Linear(8*16, 16)

# Calculating the final context vector
context_vector_2 = linear(multihead_z_2.flatten())