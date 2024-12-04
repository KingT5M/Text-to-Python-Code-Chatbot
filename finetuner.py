#IMPORT THE NECESSARY LIBRARIES.
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
import evaluate
import math 

#LOAD PREPROCESSED DATA AND MODEL
tokenized_datasets = load_from_disk("./preprocessed_datasets")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

#ADD PADDING TOKEN TO THE MODEL YOU HAD ADDED TO THE TOKENIZER
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': ['pad']})
    model.resize_token_embeddings(len(tokenizer))
    
#DEFINE MONITORING METRICS
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pad_token_id = tokenizer.pad_token_id #get the id of the padding token so that they can be ignored
    #calculate perplexity: measures model's confidence its predictions. Lower values indicate better performance  
    loss = model.compute_loss(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index = pad_token_id)
    perplexity = math.exp(loss.item()) if loss.item() < 300 else float("inf")
    return {"perplexity": perplexity}