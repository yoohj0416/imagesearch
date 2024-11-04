import os
import pandas as pd
from tqdm import tqdm
import ast

import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# Hyperparameters for gemini
genai.configure(api_key=os.environ["API_KEY"])
language_model_id = "gemini-1.5-flash"

# Hyperparameters for embedding model
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"

# Path for image captions
captions_path = '/home/hojin/imagesearch/outputs/captions-drama-firstframes-florence2.csv'

# Prompt format for extracting object counting condition
prompt_format_condition = \
    "Extract WHERE conditions which counts the number of objects from input text like example." + "\n" \
    "example 1: There is more than 5 car in the image -> car_cnt >= 5" + "\n" \
    "example 2: There is 3 pedestrian in the image -> pedestrian_cnt == 3" + "\n" \
    "Input text: "
instruction_condition = "Please generate the response in the form of a Python dictionary string with keys 'conditions' The value of 'conditions' is a list(str), of which each item is a condition. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this: {'conditions': [condition1, condition2, ...]}. If there are not counting condition make empty list. If there are multiple contions, make multiple conditions."

def main():
    
    input_text = "There are more than 5 people in the street"

    # Load embedding model
    embed_model = SentenceTransformer(embed_model_id)

    # Load language model
    language_model = genai.GenerativeModel(language_model_id)

    # Load descriptions and save embeddings
    image_metadata = pd.read_csv(captions_path)
    image_ids = image_metadata['ImageID'].tolist()

    # Compute embeddings for input text
    input_embedding = embed_model.encode(input_text, convert_to_tensor=True)

    caption_embeddings = []
    # First, I want to encode 1,000 video captions for debugging
    for i, image_id in enumerate(tqdm(image_ids[:1000])):
    # for i, image_id in enumerate(image_ids):
        splited_captions = image_metadata['Caption'][i].split('.')

        caption_embedding = embed_model.encode(splited_captions, convert_to_tensor=True)
        caption_embeddings.append(caption_embedding) 

    # Load object names and save embeddings, and make objects table
    # Remove duplicaated object names
    od_results = image_metadata['OD'].tolist()
    # Make string to list using package ast
    od_results = [ast.literal_eval(od_result) for od_result in od_results]
    # Make list one-dimensional and remove duplicated objects
    all_object_names = list(set([object_name for od_result in od_results for object_name in od_result]))
    # Extend list with add 's' to object names (if not end with 's')
    all_object_names = all_object_names + [object_name + 's' if object_name[-1] != 's' else object_name for object_name in all_object_names]
    # And sort by alphabetical order
    all_object_names.sort()
    # Make object table
    object_table = []
    for i, image_id in enumerate(image_ids[:1000]):
        od_result = od_results[i]
        # count object number and make object table as dictionary
        object_count = {}
        for od in od_result:
            if od in object_count:
                object_count[od] += 1
            else:
                object_count[od] = 1
        object_table.append(object_count)

    # Compute embeddings for object names
    object_names_embeddings = embed_model.encode(all_object_names, convert_to_tensor=True)

    # Extract object counting condition from input text using Gemini
    prompt_condition = prompt_format_condition + input_text + "\n\n" + instruction_condition
    response_condition = language_model.generate_content(prompt_condition)

    # Compute score for finding similar images

    # Scoring Step 1: Compute similarity between input text and descriptions (each sentence)
    
    # Scoring Step 2: Compute similarity between counting condition column name and object names (make similarity matrix)
    # Split response_condition to get counting condition
    print(response_condition.text)
    condition = ast.literal_eval(response_condition.text)['conditions'][0]
    conditions = condition.split(' ')
    condition_column_name = conditions[0].split('_')[0]
    condition_column_name_embedding = embed_model.encode(condition_column_name, convert_to_tensor=True)
    condition_sign = conditions[1]
    condition_number = int(conditions[2])

    similarities_object_and_condition = F.cosine_similarity(object_names_embeddings, condition_column_name_embedding, dim=1)
    similarities_object_and_condition = similarities_object_and_condition.cpu().numpy()

    # Scoring Step 3: If similarty between input text and object names is higher than threshold and satisfy counting condition, then store true
    bool_list = []

    # Get column name which has highger similarity than threshold (0.4) and the highest similarity with condition column name
    for i, image_id in enumerate(image_ids[:1000]):
        object_count = object_table[i]
        
        highest_similarity = 0
        highest_similarity_object = ''
        highest_similarity_object_cnt = 0
        for object, object_cnt in object_count.items():
            if object_cnt > 1:
                object = object + 's' if object[-1] != 's' else object
            object_similarity = similarities_object_and_condition[all_object_names.index(object)]
            if object_similarity > 0.5 and object_similarity > highest_similarity:
                highest_similarity = object_similarity
                highest_similarity_object = object
                highest_similarity_object_cnt = object_cnt

        if highest_similarity_object != '':
            if condition_sign == '>':
                if highest_similarity_object_cnt > condition_number:
                    bool_list.append(True)
                else:
                    bool_list.append(False)
            elif condition_sign == '>=':
                if highest_similarity_object_cnt >= condition_number:
                    bool_list.append(True)
                else:
                    bool_list.append(False)
            elif condition_sign == '==':
                if highest_similarity_object_cnt == condition_number:
                    bool_list.append(True)
                else:
                    bool_list.append(False)
            elif condition_sign == '<=':
                if highest_similarity_object_cnt <= condition_number:
                    bool_list.append(True)
                else:
                    bool_list.append(False)
            elif condition_sign == '<':
                if highest_similarity_object_cnt < condition_number:
                    bool_list.append(True)
                else:
                    bool_list.append(False)
            else:
                bool_list.append(False)
        else:
            bool_list.append(False)

    print(bool_list)
    print(len(bool_list))
    exit(0)

    # Compute all scores
    # If condition is False, Score: average of similarities between input text and descriptions
    # If condition is True, Score: 0.5 * average of similarities between input text and descriptions + 0.5 
    scores = []
    for i, image_id in enumerate(image_ids[:1000]):
        caption_embedding = caption_embeddings[i]
        similarities = F.cosine_similarity(input_embedding, caption_embedding, dim=1)
        scores.append(similarities.cpu().numpy().mean())

    # Find value that over threshold
    threshold = 0.3
    for i, score in enumerate(scores):
        if score > threshold:
            print(f"Image ID: {image_ids[i]}, Score: {score}")


if __name__ == "__main__":
    main()