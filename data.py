#import the necessary libraries
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

#download the dataset
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
   
#PREPROCESS THE DATA
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

