#import the necessary libraries
from datasets import load_dataset

#download and print dataset
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
print (dataset)

#check if dataset has train, validation and test splits
if 'train' in dataset:
    print("Dataset has a train split")
if 'validation' in dataset:
    print("Dataset has a validation split")
if 'test' in dataset:
    print("Dataset has a test split")
    
#preprocess the data
