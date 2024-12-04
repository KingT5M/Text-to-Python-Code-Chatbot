#IMPORT THE NECESSARY LIBRARIES.
from transformers import AutoModelForCausalLM, Autotokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
import evaluate
import math 

#LOAD PREPROCESSED DATA AND MODEL
tokenized_datasets = load_from_disk("./preprocessed_datasets")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer = Autotokenizer.from_pretrained("Salesforce/codegen-350M-mono")

#ADD PADDING TOKEN TO THE MODEL YOU HAD ADDED TO THE TOKENIZER
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': ['pad']})
    model.resize_token_embeddings(len(tokenizer))