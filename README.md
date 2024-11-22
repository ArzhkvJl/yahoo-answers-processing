# Fine-Tuning FLAN-T5-base on YahooAnswers Dataset
## Overview 
The goal of this project is to prepare a new dataset based on the YahooAnswers dataset, fine-tune the FLAN-T5-base language model on this dataset, and create some question-answer exchange with the new LLM over the terminal. 
The project includes examples of old and new answers to the same questions and the corresponding evaluation metric.

## Installation
 Clone this repository and install packages from requirements.txt.
```bash
git clone https://github.com/ArzhkvJl/yahoo-answers-processing  <my-folder>
cd <my-folder>
pip install -r requirements.txt
```
## Usage

```bash
# Start fine-tuning process and save a new model
python fine-tuning

# Ask questions to trained model 
python "fine-tuned model qa.py"
```

## 1. Preparing training dataset
There are several YahooAnswers dataset in the net. I copied one of them, added the column titles and pushed it into HuggingFace Datasets as huggingface.

## 2. Fine-tunung
The script fine-tuning.py consists steps to perform a fine-tuning.

Updating training dataset by selecting only several rows to avoid OutOfMemory exceptions. 
Data preprocessing also includes converting the dataset into a format suitable for the question-answering task.
Then loading the tokenizer, model, and data collator, setting training arguments and starting a train loop.
While this process all metrics are showing in the https://wandb.ai. But due to the lack of memory I had to implement this process without compute_metrics parameter. Instead, I used a simple function from script "bert-score.py" to check this metric for some model answers.
Output from the https://wandb.ai:
![img_2.png](img_2.png)
Loss of the two best checkpoints:
![img_4.png](img_4.png)
BERTScore evaluates the semantic similarity between the generated predictions and the references. So I add several trained models into this script and found the best checkpoint (red one on the picture).
![img_1.png](img_1.png)

After choosing the best checkpoint and comparing its answers with the answers of flan-t5 this was pushed to HuggingFace Hub as "JuliaTsk/flan-t5-base-finetuned".

Comparing answers:
![img_6.png](img_6.png)

![img_8.png](img_8.png)

![img_7.png](img_7.png)





