from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
output = tokenizer.encode("This is ")
print(output.tokens, output.ids)