from transformers import pipeline
from textIdentification import image_classifier
from textIdentification import sent_analisys

image_text_generator = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
result = image_text_generator('OIP (1).jpeg', max_new_tokens=150)

test_text = "El viaje estuvo increible, reoma es un lugar excelente para ir a visitar con tu pajera, realmente sali enamorado de ese lugar"

text_result = sent_analisys(test_text)

classifier = image_classifier(result[0]['generated_text'])

print(text)
print(classifier)


