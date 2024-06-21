from datasets import load_dataset
# mrpc dataset from glue benchmark
dataset = load_dataset('glue', 'mrpc')
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 1725
#     })
# })

print(dataset['train'])

# Dataset({
#     features: ['sentence1', 'sentence2', 'label', 'idx'],
#     num_rows: 3668
# })

###########here sentence1 , sentence2, label and idx are column name

print(dataset['train'][0])
# {'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
#  'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', 
#  'label': 1, 
#  'idx': 0}

print(dataset['train'][0:7])
# {'sentence1': [
#         'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .", 
#         'They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .', 
#         'Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .',
#         'The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .',
#         'Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .',
#         'The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .'
#         ], 
#  'sentence2': [
#         'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', 
#         "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .", 
#         "On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .", 
#         'Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .',
#         'PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .',
#         "With the scandal hanging over Stewart 's company , revenue the first quarter of the year dropped 15 percent from the same period a year earlier .", 
#         'The tech-laced Nasdaq Composite .IXIC rallied 30.46 points , or 2.04 percent , to 1,520.15 .'], 
#  'label': [1, 0, 1, 0, 1, 1, 0], 
#  'idx': [0, 1, 2, 3, 4, 5, 6]}

print(dataset['train'][0]['sentence1'])
# Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .

print(dataset['train'].features)
# {'sentence1': Value(dtype='string', id=None), 
#  'sentence2': Value(dtype='string', id=None), 
#  'label': ClassLabel(names=['not_equivalent', 'equivalent'], id=None),
#  'idx': Value(dtype='int32', id=None)}





##############tokenizing this dataset #####################
from transformers import AutoTokenizer
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], padding="max_length", truncation=True, max_length= 128
    )

tokenize_dataset = dataset.map(tokenize_function)

print(tokenize_dataset.column_names)
## ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids']