from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch


##################################Normal code with Abstraction ###########################

classifier = pipeline("sentiment-analysis")
classifier([
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
])

#classifier = [{'label': 'POSITIVE', 'score': 0.9598047137260437},
#               {'label': 'NEGATIVE', 'score': 0.9994558095932007}]






########################Now without any abstraction #################################
############################### part 1 tokenization #######################################

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_input = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
        ]

inputs = tokenizer(raw_input, padding=True, truncation= True, return_tensors="pt")

# inputs = {'input_ids': tensors([
#                                   [101, 1045, 1005, 2310, 2042, 3403, 2005,1037,17662],
#                                   [101, 1045, 5223, 2023, 2061, 2172, 999, 102, 0   ]
#                               ]),
#            'attention_maks': tensor([
#                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
#                                      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,]
#                                   )]
#           }





##################################### part 2 model calling to generate logits ###################

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
output = model(**inputs)

# output.logits = tensor([[-1.5607, 1.6123],
#                         [4.1692, -3.3464]],grad_fn=<AddmmBackward>)




#################################### part 3 postprocesssing converting logits to softmax(probability######

predictions = torch.nn.functional.softmax(output.logits, dim=-1)
# prediction = tensor([[ 4.0195e-02, 9.5980e-01],
#                      [9.9946e-01, 5.4418e-04]], grade_fn=<SoftmaxBackward>)


####################### to find which of the 1st or 2nd corresopond to positive/negative label######
print(model.config.id2label)
# {0: 'Negative', 1: 'Positive'} mens 0 index corresponds to Negative while 1 index corresponds to Positive
