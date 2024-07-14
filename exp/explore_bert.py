from transformers import BertTokenizer
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Token IDs you want to decode
token_ids = [ 101, 3793, 9780,  102]

# Decode the token IDs to text
tokens = tokenizer.decode(token_ids)

print(tokens)


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
model = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')

# Tokenize the word "culture"
inputs = tokenizer("culture", return_tensors="pt")

# Get embeddings from BERT
with torch.no_grad():
    outputs = model(**inputs)
    # Use the last hidden state
    embeddings = outputs.last_hidden_state

# Assume we want to use the embedding of the first token ([CLS] token) for simplicity
# You might want to use a different strategy depending on your use case
embedding_vector = embeddings[:, 0, :]

# Define a linear layer to map from the BERT embedding dimension to 2
linear_layer = nn.Linear(in_features=256, out_features=2)  # BERT model has a hidden size of 256

# Apply the linear layer to the embedding vector
output_vector = linear_layer(embedding_vector)

print(output_vector)