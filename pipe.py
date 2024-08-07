from transformers import pipeline

# ? Generacion de texto
classifier = pipeline("text-generation", model="distilgpt2")
#Le podemos pasar un modelo el cual queremos usar

res = classifier("Know i will talk about cars, a car is",max_length=60,)

print(res)

classifier = pipeline("sentiment-analysis")

res = classifier("I think ITS PERFECT")

#Se le pueden pasar textos multiples
print(res)

# ! LO QUE BUSCO YO

classifier = pipeline("zero-shot-classification")

result = classifier("Im here seeing the Colosseum, it's incredible.",
  
  candidate_labels=["Rome", "Paris", "Spain"]
)

print(result)


pipe = pipeline("translation", model="facebook/nllb-200-3.3B")

result = pipe("translate English to Spanish: Hello, how are you")
print(result)

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Crea el tokenizer y el modelo
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-large")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-large")

# Define la entrada en el formato adecuado
input_text = "translate English to Spanish: Hello, how are you"

# Genera la traducción
translated_text = model.generate(input_text, max_length=50, num_return_sequences=1)

# Decodifica el resultado
decoded_text = tokenizer.decode(translated_text[0], skip_special_tokens=True)
print("Traducción:", decoded_text)
