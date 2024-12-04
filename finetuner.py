#IMPORT THE NECESSARY LIBRARIES.
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
import torch
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

#DEFINE TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir = "./codegen-350M-mono-finetuned", #location weights, logs and model will be stored
    per_device_eval_batch_size = 4, #Batch size for training on each device
    per_device_eval_batch_size = 4, #Batch size for evaluation on each device
    gradient_accumulation_steps = 4, #Accumulate gradients over multiple steps to simulate a larger batch size
    evaluation_strategy = "steps", #Evaluate model after a defined number of steps
    eval_steps = 500, #Number of steps between each evaluation.
    save_steps = 500, #Save model after a defined number of steps.
    logging_steps = 500, #Log training metrics after a defined number of steps.(should be similar to eval steps)
    learning_rate = 2e-5, #Initial learning rate for training.
    weight_decay = 0.01, #regularization to prevent overfitting
    warmup_steps = 500, #Warmup steps to stabilize learning rate at the beginning of training.
    num_train_epochs = 3, #number of times the model will pass through the training dataset.
    fp16 = False, #we have no cpu
    report_to = "tensorboard", #Log metrics to TensorBoard for monitoring.
    push_to_hub = False, #no need to push model checkpoints to huggingface since we are doing it offline
    load_best_model_at_end = True, #load the best model at the end of training
    metric_for_best_model = "perplexity", #Metric to determine the best model (lower perplexity is better).
    greater_is_better = False #ensures lower perplexity signifies best model
)

#INITIALIZE TRAINER
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)