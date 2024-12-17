import tiktoken
from transformers import GPT2TokenizerFast

encoding = tiktoken.encoding_for_model("gpt-4")
# or "gpt-3.5-turbo" or "text-davinci-003"

text_to_token = input("Enter text: ")
tokens = encoding.encode(text_to_token)
token_count = len(tokens)

print(f"Token count: {token_count}")
print(tokens)

# set the gpt4 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4')
tokens = tokenizer.encode(text_to_token)
print(tokens)
