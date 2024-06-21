from transformers import Pipeline

############ Sentiment Analysis #############

classifier = Pipeline('sentiment-analysis')
output = classifier("I've waiting for a Hugging Face course my whole life.")
# output=  [{'label':'POSITIVE','score':0.959804}]

### passing multiple text to same pipeline
output = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life",
        "I hate this so much!",
    ]
)
# output= [{'label': 'POSITIVE', 'score': 0.959804},{'label':'NEGATIVE', 'score': 0.9994558}]






##################### Zero-Shot-Classification ################
classifier = Pipeline("zero-shot-classification")
output = classifier(
    "This is a course about Transformer library",
    candidate_labels=['education','politics','business'],
)
# output ={'sequence': 'This is a course about the Transformers library',
#           'labels':['education', 'business', 'politics'],
#           'score': [0.844596, 0.111976, 0.04342744]}




##################### text-generation ####################
generator = Pipeline('text-generation',model='distilgpt2')
output = generator(
    "In this course, we will teach you how to",
    max_length = 30,
    num_return_sequence = 2,
)

# output = [{'generated_text': 'In this course, we will teach you how to think about the idea of the ideal of a real-world natural person as a animal.'},
#           {'generated_text': 'In this course, we will teach you how to use them to work on your own project. This course is designed to support both those with experience and'}
#          ]



##################### fill-mask ##########################
unmasker = Pipeline('fill-mask')
output = unmasker('This course will teach you all about <mask> models.',top_k=2)
# output = [{'sequence': 'This course will teach you all about mathematical models',
#            'score': 0.196198,
#            'token': 30412,
#            'token_str': 'mathematical'},
#           
#            {'sequence': 'This course will teach you all about computational models',
#            'score': 0.0496198,
#            'token': 38112,
#            'token_str': 'computational'},
#           
#            ]


####################### ner #################################
ner = Pipeline("ner", grouped_entities=True)
output = ner("My name is Anurag and I work at JellyFish in Noida")
# output =[
#         {'entity_group': 'PER', 'score': 0.99816, 'word': 'Anurag', 'start':11, 'end': 18},
#         {'entity_group': 'ORG', 'score': 0.97960, 'word': 'JellyFish', 'start':33, 'end':45},
#         {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Noida', 'start':49, 'end':57}]



##################### summarization #################################
summarizer = Pipeline('summarization')
output = summarizer(""" 
                    Johannes Gutenberg (1398 – 1468) was a German goldsmith and publisher who introduced printing to Europe. His introduction of mechanical movable type printing to Europe started the Printing Revolution and is widely regarded as the most important event of the modern period. It played a key role in the scientific revolution and laid the basis for the modern knowledge-based economy and the spread of learning to the masses.

                    Gutenberg many contributions to printing are: the invention of a process for mass-producing movable type, the use of oil-based ink for printing books, adjustable molds, and the use of a wooden printing press. His truly epochal invention was the combination of these elements into a practical system that allowed the mass production of printed books and was economically viable for printers and readers alike.

                    In Renaissance Europe, the arrival of mechanical movable type printing introduced the era of mass communication which permanently altered the structure of society. The relatively unrestricted circulation of information—including revolutionary ideas—transcended borders, and captured the masses in the Reformation. The sharp increase in literacy broke the monopoly of the literate elite on education and learning and bolstered the emerging middle class.
                    """)
# output = [{"summariy_text":"The German Johannes Gutenberg introduced printing in Europe. His invention had a decisive contribution in spread of mass-learning and in building the basis of the modern society. Gutenberg major invention was a practical system permitting the mass production of printed books. The printed books allowed open circulation of information, and prepared the evolution of society from to the contemporary knowledge-based economy."}]




####################### translation ######################################
translator = Pipeline('translation',model="Helsinki-NLP/opus-mt-fr-en")
output = translator("Ce cours est produit par Hugging Face.")
# output = [{'translaton_text':'This is course is produced by Hugging Face.'}]

