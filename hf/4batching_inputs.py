from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Let's try to tokenize!")
print(inputs["input_ids"])
# [101, 2292, 1005, 3046, 2000, 19204, 4697, 999, 102]

print(tokenizer.decode(inputs["input_ids"]))
# "[CLS] Let's try to tokenize! [SEP]"


