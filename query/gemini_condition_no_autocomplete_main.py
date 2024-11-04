import os, json
# from app import app
# from flask import Flask, jsonify, request, redirect, render_template, url_for
# import duckdb
import pandas as pd
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ast

import uvicorn
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.routing import URL
from fastapi.staticfiles import StaticFiles

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


app = FastAPI()
templates_dir = "/home/hojin/imagesearch/query/templates"
templates = Jinja2Templates(directory=templates_dir)
# templates = Jinja2Templates(directory="templates")
static_dir = "/home/hojin/imagesearch/query/static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/static", StaticFiles(directory="/pool0/data_archive_hojin/drama_data"), name="static")

# Hyperparameters for embedding model
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
similarity_threshold = 0.55
top_similar_images = 300

# Hyperparameters for gemini
genai.configure(api_key=os.environ["API_KEY"])
language_model_id = "gemini-1.5-flash"

# Prompt format for extracting object counting condition
prompt_format_condition = \
    "Extract WHERE conditions which counts the number of objects from input text like example." + "\n" \
    "example 1: There is more than 5 car in the image -> car_cnt >= 5" + "\n" \
    "example 2: There is 3 pedestrian in the image -> pedestrian_cnt == 3" + "\n" \
    "Input text: "
instruction_condition = "Please generate the response in the form of a Python dictionary string with keys 'conditions' The value of 'conditions' is a list(str), of which each item is a condition. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this: {'conditions': [condition1, condition2, ...]}. If there are not counting condition make empty list. If there are multiple contions, make multiple conditions."

# Number for debugging
debug_num = 1000

# Path to image captions
captions_path = '/home/hojin/imagesearch/outputs/captions-drama-firstframes-florence2.csv'
# images_base = 'firstframes'
images_base = 'firstframes_jpg'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Make logging file
logs_dir = "/home/hojin/imagesearch/query/logs"
if not os.path.exists(logs_dir):
	os.makedirs(logs_dir)
# if not os.path.exists('logs'):
	# os.makedirs('logs')
log_file_name = 'event_wo_autocomplete.log'
log_file_path = os.path.join(logs_dir, log_file_name)
# log_file_path = os.path.join('logs', log_file_name)
if not os.path.exists(log_file_path):
	with open(log_file_path, 'w') as f:
		f.write('')

global embed_model
global language_model
global image_ids
global caption_embeddings
global image_metadata
global all_object_names
global object_table
global object_names_embeddings

def get_image_ids_from_search_term(model, text, ids, embbedings, vector_dim=512, threshold=0.5, top_k=300):

	# Compute similarity between search term and video descriptions
	search_embedding = model.encode([text], convert_to_tensor=True)
	search_embedding = F.layer_norm(search_embedding, normalized_shape=(search_embedding.shape[1],))
	search_embedding = search_embedding[:, :vector_dim]
	search_embedding = F.normalize(search_embedding, p=2, dim=1)

	# Get image ids whose cosine similarity is above the threshold from image_ids
	similarities = F.cosine_similarity(search_embedding, embbedings, dim=1)
	similarities = similarities.cpu().numpy()
	similar_image_ids = [ids[i] for i in np.where(similarities > threshold)[0]]

	# Sort image ids by cosine similarity
	similar_image_ids = sorted(similar_image_ids, key=lambda x: similarities[ids.index(x)], reverse=True)

	# Return top-k image paths
	return similar_image_ids[:top_k]


def init():

	global embed_model
	global language_model
	global image_ids
	global caption_embeddings
	global image_metadata
	global all_object_names
	global object_table
	global object_names_embeddings

	# Load embedding model for similarity search
	embed_model = SentenceTransformer(embed_model_id)

	# Load language model
	language_model = genai.GenerativeModel(language_model_id)

	# Load video captions and make it as embedding vectors
	print('Encoding video descriptions...')
	image_metadata = pd.read_csv(captions_path)

	start_time = time.time()

	image_ids = image_metadata['ImageID'].tolist()
	caption_embeddings = []
	# First, I want to encode part of video captions for debugging
	for i, image_id in enumerate(tqdm(image_ids[:debug_num])): 
	# for i, image_id in enumerate(tqdm(image_ids)):
		caption_embeddings.append(embed_model.encode(image_metadata['Caption'][i], convert_to_tensor=True))
	caption_embeddings = torch.cat(caption_embeddings, dim=0)
	caption_embeddings = torch.reshape(caption_embeddings, (len(image_ids[:debug_num]), -1))

	# Load object names and save embeddings, and make objects table
	od_results = image_metadata['OD'].tolist()
	od_results = [ast.literal_eval(od_result) for od_result in od_results]
	all_object_names = list(set([object_name for od_result in od_results for object_name in od_result]))
	all_object_names = all_object_names + [object_name + 's' if object_name[-1] != 's' else object_name for object_name in all_object_names]
	all_object_names.sort()

	object_table = []
	for i, image_id in enumerate(image_ids[:debug_num]):
		od_result = od_results[i]
		object_count = {}
		for od in od_result:
			if od in object_count:
				object_count[od] += 1
			else:
				object_count[od] = 1
		object_table.append(object_count)

	object_names_embeddings = embed_model.encode(all_object_names, convert_to_tensor=True)

	print('Time to encode video descriptions: ', (time.time() - start_time), 's')

@app.get('/')
async def upload_form(request: Request):
	global image_ids

	image_count = len(image_ids)
	formatted_image_count = "{:,}".format(image_count)

	return templates.TemplateResponse('no_autocomplete.html', {"request": request, "image_count": formatted_image_count})

# @app.post("/get_list")
@app.get("/get_list")
async def get_list(text: str):

	global embed_model
	global language_model
	global image_ids
	global caption_embeddings
	global all_object_names
	global object_table
	global object_names_embeddings

	print('Search term for image search: ', text)

	if text != '':
		start_time = time.time()

		input_embedding = embed_model.encode(text, convert_to_tensor=True)
		print(input_embedding.shape, caption_embeddings.shape)
		caption_similarities = F.cosine_similarity(input_embedding, caption_embeddings, dim=1)
		
		# Extract object counting condition from input text using Gemini
		prompt_condition = prompt_format_condition + text + "\n\n" + instruction_condition
		response_condition = language_model.generate_content(prompt_condition)

		condition = ast.literal_eval(response_condition.text)['conditions'][0]
		conditions = condition.split(' ')
		condition_column_name = conditions[0].split('_')[0]
		condition_column_name_embedding = embed_model.encode(condition_column_name, convert_to_tensor=True)
		condition_sign = conditions[1]
		condition_number = int(conditions[2])

		similarities_object_and_condition = F.cosine_similarity(object_names_embeddings, condition_column_name_embedding, dim=1)
		similarities_object_and_condition = similarities_object_and_condition.cpu().numpy()

		object_similarity_list = []
		for i, image_id in enumerate(image_ids[:debug_num]):
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
						object_similarity_list.append(highest_similarity)
					else:
						object_similarity_list.append(.0)
				elif condition_sign == '>=':
					if highest_similarity_object_cnt >= condition_number:
						object_similarity_list.append(highest_similarity)
					else:
						object_similarity_list.append(.0)
				elif condition_sign == '==':
					if highest_similarity_object_cnt == condition_number:
						object_similarity_list.append(highest_similarity)
					else:
						object_similarity_list.append(.0)
				elif condition_sign == '<=':
					if highest_similarity_object_cnt <= condition_number:
						object_similarity_list.append(highest_similarity)
					else:
						object_similarity_list.append(.0)
				elif condition_sign == '<':
					if highest_similarity_object_cnt < condition_number:
						object_similarity_list.append(highest_similarity)
					else:
						object_similarity_list.append(.0)
			else:
				object_similarity_list.append(.0)

		# Combine caption similarities and object similarities with weight (each 0.5)
		scores = 0.5 * caption_similarities + 0.5 * torch.tensor(object_similarity_list).to(device)

		# Sort image ids by similarity score
		scores = scores.cpu().numpy()
		similar_image_ids = [image_ids[i] for i in np.argsort(scores)[::-1]]

		print('Time to search images: ', (time.time() - start_time), 's')

		# Return top-k image paths
		# return [os.path.join('static', images_base, item + '.png') for item in similar_image_ids[:top_similar_images]]
		return [os.path.join('static', images_base, item + '.jpg') for item in similar_image_ids[:top_similar_images]]
		# return []

	else:
		return []

@app.get("/log_key_event")
async def log_key_event(request: Request, text: str):

	client_ip = request.client.host
	current_time = datetime.now().isoformat()

	with open(log_file_path, 'a') as f: 
		f.write(f'Client IP: {client_ip}, Text: {text}, Time: {current_time}, Key event\n')

	print(f'Client IP: {client_ip}, Text: {text}, Time: {current_time}, Key event')
	
	return {"message": "Log saved successfully"}

@app.get("/log_clear_event")
async def log_clear_event(request: Request, text: str):

	client_ip = request.client.host
	current_time = datetime.now().isoformat()

	with open(log_file_path, 'a') as f:
		f.write(f'Client IP: {client_ip}, Text: {text}, Time: {current_time}, Clear button event\n')

	print(f'Client IP: {client_ip}, Text: {text}, Time: {current_time}, Clear button event')

	return {"message": "Log cleared successfully"}


if __name__ == "__main__":
    init()
    # uvicorn.run(app, port=8080)
    uvicorn.run(app, host='0.0.0.0', port=8080)
    # uvicorn.run(app, host='0.0.0.0', port=80)