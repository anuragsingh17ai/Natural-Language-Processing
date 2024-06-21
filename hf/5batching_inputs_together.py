from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentences= [
        "I've waiting for a Hugging Face course my whole life.",
        "I hate this.",
        ]

tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
print(tokens)


ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
print(ids)

################## till here even though we tokenize this sentence into tokens
################# there after we map it to it's corresponding id to model tokenizer vocablry
################ it wil be hard to convert it into tensors as both tokenize id's are not of same length


# so we can combine tokenize and id conversion part into 1 line and if we did padding = true.it will
# make token id list of same length so, it can be converted into tensors


inputs = tokenizer(sentences, padding = True)
print(inputs)
