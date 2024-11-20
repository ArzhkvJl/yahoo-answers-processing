from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

input_text = "Who you call a person that carries a golfers club?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
cleaned_output = re.sub(r"<.*?>", "", tokenizer.decode(outputs[0])).strip()
print("Question:", input_text)
print("Answer:",cleaned_output)
input_text = "What are  the characteristics of a leader?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
cleaned_output = re.sub(r"<.*?>", "", tokenizer.decode(outputs[0])).strip()
print("Question:", input_text)
print("Answer:", cleaned_output)
