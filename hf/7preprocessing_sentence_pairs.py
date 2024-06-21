from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenizer("My name is Sylvain","I work at Hugging Face.")

###{
###  'input_ids':[101, 2026, 2171, 2003, 25353, 22144, 2378, 1012, 102, 1045, 2147, 2012, 17662, 2227, 1012, 102],
###  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,],
### 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1 ] }

input = tokenizer(
    ["My name is Sylvain.", "Going to the cinema."],
    ["I work at Hugging Face,","This movie is great"],
    padding = True
)
print(input)

# {
#    'input_ids': [
#                        [101, 2026, 2171, 2003, 25353, 22144, 2378, 1012, 102, 1045, 2147, 2012, 17662, 2227, 1010, 102],
#                        [101, 2183, 2000, 1996, 5988, 1012, 102, 2023, 3185, 2003, 2307, 102, 0, 0, 0, 0]], 
#    'token_type_ids': [
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
#                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]], 
#    'attention_mask': [
#                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
#                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
#    }