def ordenar_scores(tags_desordenados, tags_ordenados, scores):
    # Crea un diccionario que asocie cada tag con su score
    tag_score_dict = dict(zip(tags_desordenados, scores))
    
    # Ordena los scores en el mismo orden que los tags ordenados
    sorted_scores = [tag_score_dict[tag] for tag in tags_ordenados]
    
    return sorted_scores

# Ejemplo de uso
tags_images = [
    "Landscape", "City", "Beach", "Mountain", "Forest",
    "Sunset", "Night", "Day", "Buildings", "Calm"
]

unordered_tags = ['Night', 'Sunset', 'Buildings', 'City', 'Landscape', 'Mountain', 'Beach', 'Day', 'Forest', 'Calm']

unorderd_tags_scores = [0.5859789848327637, 0.08152350783348083, 0.07012838870286942, 0.05843888968229294, 0.04835524037480354, 0.03957007825374603, 0.03907245397567749, 0.03371385484933853, 0.027286037802696228, 0.01593264937400818]

sorted_scores = ordenar_scores(unordered_tags, tags_images, unorderd_tags_scores)
print(sorted_scores)
