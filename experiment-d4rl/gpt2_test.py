from transformers import GPT2Tokenizer, GPT2Model
from transformers import pipeline
import torch
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
#pipeline = pipeline('feature-extraction', model=model)


text = "These are context:"
############################ TOKENIZER test
def token_text(text):
    return tokenizer.tokenize(text)

def token_id(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)

text = "These are context:"
tokens = token_text(text)
token_ids = token_id(tokens)
print(text, ' ',tokens , ' ', )

# text = "Context ends, now predict:"
# tokens = token_text(text)
# print(text, ' ',tokens , ' ', token_id(tokens))

############################# embedding test
# data = pipeline("this")
# print(f"{data.shape=}")
from transformers import pipeline
# pipeline = pipeline('feature-extraction',fromework="pt", model=model, tokenizer=tokenizer)
# text = "These are context:"
# token_embeddings = pipeline(text)
# print(text)
# print(f"{len(token_embeddings)=}, {len(token_embeddings[0])=}, {len(token_embeddings[0][0])=}")

# print("Output of pipeline")
# print(token_embeddings[0][0][:10])

# print("Output of self indexing")
# word_embeddings = model.get_input_embeddings()#.weight
# print(f"{word_embeddings.weight.shape=}")

# data = np.array([token_ids[0]])
# data = torch.LongTensor(data)
# out = torch.LongTensor(data)
# print(word_embeddings(out).shape)
# print(word_embeddings(out)[:, :10])

#encoded_input = tokenizer(text, return_tensors='pt')
"""
input_ids, token_type_ids, attention_mask
"""
#print(encoded_input)

# tokens = tokenizer.tokenize(text)
# print(f"{tokens=}")
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(f"{ids=}")

t = "0.123456789"
c = token_text(t)
print(f"{t}, {c=}, {len(c)}")

t = "0.7734629"
c = token_text(t)
print(f"{t}, {c=}, {len(c)}")

t = "0.1"
c = token_text(t)
print(f"{t}, {c=}, {len(c)}")

t = "-1.775562"
c = token_text(t)
print(f"{t}, {c=}, {len(c)}")