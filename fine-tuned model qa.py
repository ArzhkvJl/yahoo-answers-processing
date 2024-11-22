import re
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration

# Load the tokenizer, model
MODEL_NAME = "JuliaTsk/flan-t5-base-finetuned"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Create the user primpt
print("Type a question: ")
input_text = str(input())
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate and print an answer
outputs = model.generate(input_ids)
cleaned_output = re.sub(r"<.*?>", "", tokenizer.decode(outputs[0])).strip()
print("Question:", input_text)
print("Answer:", cleaned_output)