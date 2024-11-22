import matplotlib.pyplot as plt
import evaluate
import re
from datasets import load_dataset, DatasetDict
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "JuliaTsk/yahoo-answers"
NUM_OF_QUESTIONS = 10


def preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    prefix = "Please answer this question: "
    # The "inputs" are the tokenized questions from dataset:
    inputs = [prefix + qn for qn in examples["question title"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # The "labels" are the tokenized outputs:
    labels = tokenizer(text_target=[str(ans) if ans is not None else "" for ans in examples["best answer"]],
                       max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Acquire the training data from Hugging Face
data_files = {"train": "train.csv", "test": "test.csv"}
yahoo_answers_qa = load_dataset(DATASET_NAME, data_files=data_files)
train_subset = yahoo_answers_qa['train'].select(range(NUM_OF_QUESTIONS))
test_subset = yahoo_answers_qa['test'].select(range(NUM_OF_QUESTIONS))
yahoo_answers_qa = DatasetDict({
    "train": train_subset,
    "test": test_subset
})

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
last_checkpoint = "output/checkpoint-1500"
finetuned_model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
tokenized_dataset = yahoo_answers_qa.map(preprocess_function, batched=True)

model_ans = []
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(NUM_OF_QUESTIONS))

for i in range(NUM_OF_QUESTIONS):
    input = tokenizer(small_eval_dataset[i]['question title'], return_tensors="pt")
    output = finetuned_model.generate(**input)
    answer = re.sub(r"<.*?>", "", tokenizer.decode(output[0])).strip()
    model_ans.append(answer)

bertscore = evaluate.load("bertscore")
results = bertscore.compute(predictions=model_ans, references=small_eval_dataset['best answer'], lang="en")

# Data for F1 scores for two best trainings with the minimum loss
results = {
    'f1_first': [
        0.8338123559951782, 0.8283671140670776, 0.7786645889282227,
        0.8045203685760498, 0.7728509306907654, 0.8445399403572083,
        0.8174916505813599, 0.7821820974349976, 0.8080450892448425,
        0.7944367527961731
    ],
    'f1_second': [0.8228160738945007,
                  0.8499919772148132,
                  0.8562086224555969,
                  0.8371381163597107,
                  0.7927619218826294,
                  0.796983003616333,
                  0.8674603700637817,
                  0.850703239440918,
                  0.8529340028762817,
                  0.8319612145423889]
}

# Number of questions
questions = [f"Q{i + 1}" for i in range(len(results['f1']))]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(questions, results['f1'], marker='o', linestyle='-', color='b')
plt.plot(questions, results['n1'], marker='o', linestyle='-', color='r')

# Enhancing the plot
plt.title("F1 Scores for Each Question", fontsize=14)
plt.xlabel("Questions", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.ylim(0.7, 0.9)
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()

# Display the plot
plt.show()
