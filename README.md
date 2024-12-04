# Text-to-Python Code Chatbot

This project develops a chatbot that generates Python code from natural language instructions. It leverages the `Salesforce/codegen-350M-mono` pre-trained model and fine-tunes it on a preprocessed version of the `iamtarun/python_code_instructions_18k_alpaca` dataset. The chatbot uses the Hugging Face `transformers` library for model handling and training.

## Features

* Understands natural language instructions describing desired Python code functionality.
* Generates Python code snippets based on these instructions.
* Fine-tuned for improved performance on the target code generation task.
* Employs perplexity as the key metric to monitor and evaluate model performance during fine-tuning.
* Uses a CPU-efficient training configuration to accommodate resource limitations.
* Includes TensorBoard integration for visualizing training metrics.

## Implementation Details

### Data Preprocessing (`data.py`)

The original dataset is preprocessed to add padding tokens ("[PAD]") and then saved to disk.  Tokenization is handled using the `AutoTokenizer` from the `transformers` library, ensuring consistent tokenization between preprocessing and model fine-tuning.

### Model Fine-tuning (`finetuner.py`)

The `Salesforce/codegen-350M-mono` model is loaded using `AutoModelForCausalLM`.  Crucially, the tokenizer is also initialized, and if needed, the padding token is explicitly added to both the tokenizer and the model to handle variable sequence lengths. A custom `compute_metrics` function calculates perplexity, excluding padding tokens for accurate evaluation.  The `Trainer` from the `transformers` library manages the fine-tuning process with `TrainingArguments` optimized for a CPU environment.  TensorBoard integration is enabled for training visualization.  The fine-tuned model and updated tokenizer are saved for later use in code generation.

## Usage

The `prompt.py` script demonstrates how to use the fine-tuned model to generate Python code. A text generation pipeline is set up.  The user can provide a natural language instruction, and the pipeline will utilize the model and updated tokenizer for code generation.  

## Future Enhancements

* Incorporate a learning rate scheduler for potentially improved training efficiency.
* Experiment with different pre-trained models or model sizes.
* Explore advanced generation parameters (`temperature`, `top_p`, beam search, etc.) for controlling output diversity and quality.
* Develop a user interface to interact with the chatbot.
* Deploy the model for interactive code generation.

## Author

Ian Mwaniki Kanyi (King T5M)
