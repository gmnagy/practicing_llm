from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("This is an example of the bert tokenizer")
# tokenizer_output = tokenizer.tokenize("cmd")
print(tokens)
# ['this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '#\#izer']

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
# [2023, 2003, 2019, 2742, 1997, 1996, 14324, 19204, 17629]

token_ids = tokenizer.encode("This is an example of the bert tokenizer")
print(token_ids)
# [101, 2023, 2003, 2019, 2742, 1997, 1996, 14324, 19204, 17629, 102]

tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
# ['[CLS]', 'this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '#\#izer', '[SEP]']