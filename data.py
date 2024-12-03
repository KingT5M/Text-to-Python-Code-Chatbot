#IMPORT THE NECESSARY LIBRARIES
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np

#DOWNLOAD THE DATASET
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")

#PREPROCESS THE DATA

#Create a "text" column to keep the feature engineering logic together
def format_examples(examples):
    instruction = examples['instruction']
    input_text = examples['input'] if 'input' in examples else "" #to handle missing input values
    output_code = examples['output']
    #Concatenate using special tokens to clearly distinguish the instruction, input, and output
    prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Output:\n{output_code}"
    return {"text": prompt}
dataset = dataset.map(format_examples)

#create the train, validation and test splits
dataset_train_test = dataset['train'].train_test_split(test_size=0.2, seed = 42) #20% for testing. Seed ensures that every time you run the code, the data will be shuffled in the same way.
dataset_train_validation = dataset['train'].train_test_split(test_size=0.1, seed = 42)#8% for validation

#reconstruct the DatasetDict
dataset = DatasetDict({
    'train': dataset_train_test['train'],
    'validation': dataset_train_validation['test'],
    'test': dataset_train_test['test']
})

#check if dataset has train, validation and test splits
if 'train' in dataset:
    print("Dataset has a train split")
if 'validation' in dataset:
    print("Dataset has a validation split")
if 'test' in dataset:
    print("Dataset has a test split")
    
#print dataset
print (dataset)

#tokenize the data
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono") #choose tokenizer trained on data with python vocabulary
#first analyse the tokenization in order to choose an appropriate max length
# Add a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Add the padding token
# Now you MUST set a max_length based on your data analysis. I'm setting an example value of 512.
# YOU MUST REPLACE THIS WITH AN APPROPRIATE VALUE BASED ON YOUR LENGTH ANALYSIS
max_length = 512
def tokenize_function(examples):
    return tokenizer(examples['text'], padding = 'max_length', truncation = True, max_length = max_length)
tokenized_datasets = dataset.map(tokenize_function, batched = True)
all_lengths = [len(x) for dataset_split in tokenized_datasets.values() for x in dataset_split['input_ids']]
print(f"Max length: {max(all_lengths)}")  #Print the maximum tokenized sequence length
print(f"95% percentile: {np.percentile(all_lengths, 95)}") #Provides a good estimate for how to choose max length parameter
