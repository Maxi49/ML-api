from transformers import pipeline
from googletrans import Translator
import numpy as np

# Inicializar el traductor y el clasificador
translator = Translator()

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

image_text_generator = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Etiquetas para clasificar imágenes y textos
image_tags = [
    "Landscape", "City", "Beach", "Mountain", "Forest",
    "Sunset", "Night", "Day", "Buildings", "Calm"
]

text_tags = [
    "History", "Science", "Fantasy", "Adventure", "Romance",
    "Inspiring", "Sad", "Humorous", "Serious", "Narrative"
]

def sort_scores(unordered_tags, sorted_tags, vector):
    # Crea un diccionario que asocie cada tag con su score
    tag_score_dict = dict(zip(unordered_tags, vector))
    
    # Ordena los scores en el mismo orden que los tags ordenados
    sorted_scores = [tag_score_dict[tag] for tag in sorted_tags]
    
    return sorted_scores

# ? Funcion para traducir texto de cualquier idioma a ingles
def translate_text(text: str) -> str:
    translated_text = translator.translate(text, dest='en')
    return translated_text.text

# ? Funcion para clasificar texto dependiendo lo que querramos 
def classify_text(text: str, candidate_labels: list) -> dict:
    return classifier(text, candidate_labels=candidate_labels)

# ? Funcion para generar texto en base a imagen
def image_description_generator(img: str) -> str:
  return image_text_generator(img)

def preferences_vector_multiplication(desc_text: str, img: str):
    # !Generamos la descripcion de la imagen
    text_img = image_description_generator(img)

    # !Traducimos el texto
    desc_translated_text = translate_text(desc_text)
    
    # !Clasificamos la descripcion de la imagen del post
    img_text_classifier_result = classify_text(text_img[0]['generated_text'], image_tags)
    

    # !Clasificamos el texto de la descripcion del post
    description_text_classifier_result = classify_text(desc_translated_text, text_tags)
    
    image_unordered_tags_classification = img_text_classifier_result['labels']
    text_unordered_tags_classification = description_text_classifier_result['labels']

    image_vector_classification = img_text_classifier_result['scores']
    text_vector_classification = description_text_classifier_result['scores']

    # ! Devolvemos el orden del vector segun el array de tags original
    image_sorted_vector = sort_scores(unordered_tags=image_unordered_tags_classification , sorted_tags= image_tags, vector= image_vector_classification)
    text_sorted_vector = sort_scores(unordered_tags= text_unordered_tags_classification, sorted_tags= text_tags, vector= text_vector_classification)

    multiplied_scores = np.multiply(image_sorted_vector, text_sorted_vector)
    unitary_vector = multiplied_scores / np.linalg.norm(multiplied_scores)

    return unitary_vector

