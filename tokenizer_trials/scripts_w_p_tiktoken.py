import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
# or "gpt-3.5-turbo" or "text-davinci-003"

tokens = encoding.encode("Here is some text")
token_count = len(tokens)

print(f"Token count: {token_count}")
print(tokens)