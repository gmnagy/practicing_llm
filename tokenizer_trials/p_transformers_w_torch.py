import torch
import numpy as np
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# get the embedding vector for the word "example"
example_token_id = tokenizer.convert_tokens_to_ids(["example"])[0]
example_embedding = model.embeddings.word_embeddings(torch.tensor([example_token_id]))

print(np.linalg.norm(example_embedding.detach().numpy(), ord=2))
# print(example_embedding.shape)
# torch.Size([1, 768])


king_token_id = tokenizer.convert_tokens_to_ids(["king"])[0]
king_embedding = model.embeddings.word_embeddings(torch.tensor([king_token_id]))

queen_token_id = tokenizer.convert_tokens_to_ids(["queen"])[0]
queen_embedding = model.embeddings.word_embeddings(torch.tensor([queen_token_id]))

cos = torch.nn.CosineSimilarity(dim=1)
similarity = cos(king_embedding, queen_embedding)
print(similarity[0])
# 0.6469

similarity = cos(example_embedding, queen_embedding)
print(similarity[0])
# 0.2392