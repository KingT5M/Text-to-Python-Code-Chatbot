#import the necessary libraries
from datasets import load_dataset

#download and print dataset
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
print (dataset[dataset['train'][0]])