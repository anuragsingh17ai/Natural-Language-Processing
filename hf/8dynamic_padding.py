from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


########################## Fixed padding examples #################################################################
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"],padding="max_length",truncation=True, max_length=128
    )

tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["idx","sentence1","sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
tokenized_datasets = tokenized_datasets.with_format("torch")

