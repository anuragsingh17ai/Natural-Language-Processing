from transformers import AutoConfig, BertConfig, BertModel

######simply exchange bert-based-cased with something else
#### eg. AutoConfig.from_pretrained('gpt2')
###### config file held information about how a particular model should 
###### be instantiate 

bert_config = AutoConfig.from_pretrained('bert-based-cased')

### or we can get lower level control by
bert_config = BertConfig.from_pretrained("bert-based-cased")

##### Training code
location = 'curent_dir'
bert_model = BertModel(bert_config)
bert_model.save_pretrained(location)

##### Reloading a saved model
bert_model = BertModel.from_pretrained("my-bert-model")

