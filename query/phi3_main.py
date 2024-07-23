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

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/static", StaticFiles(directory="/pool0/data_archive_hojin/drama_data"), name="static")

# Hyperparameters for embedding model
embed_model_id = "nomic-ai/nomic-embed-text-v1.5"
matryoshka_dim = 512
similarity_threshold = 0.55
top_similar_images = 300

# Hyperparameters for phi3
language_model_id = "microsoft/Phi-3-mini-4k-instruct"
init_msg = "Make autocompletion of search phrase to find following scenario from database based on caption and object detection result."
top_autocomplete_images = 30
max_new_tokens = 1024

# Path to image captions
captions_path = '/home/hojin/imagesearch/outputs/captions-drama-firstframes-florence2.csv'
# images_base = 'firstframes'
images_base = 'firstframes_jpg'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global embed_model
global image_ids
global caption_embeddings
global language_model
global tokenizer
global image_metadata

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
	global image_ids
	global caption_embeddings
	global language_model
	global tokenizer
	global image_metadata

	# Load embedding model for similarity search
	embed_model = SentenceTransformer(embed_model_id, trust_remote_code=True)

	# Load language model for generating autocompletion
	tokenizer = AutoTokenizer.from_pretrained(language_model_id)
	language_model = AutoModelForCausalLM.from_pretrained(
		language_model_id,
		device_map=device,
		torch_dtype="auto",
		trust_remote_code=True,
	)

	# Load video captions and make it as embedding vectors
	print('Encoding video descriptions...')
	image_metadata = pd.read_csv(captions_path)

	start_time = time.time()

	image_ids = image_metadata['ImageID'].tolist()
	caption_embeddings = []
	# First, I want to encode 1,000 video captions for debugging
	for i, image_id in enumerate(tqdm(image_ids[:1000])): 
	# for i, image_id in enumerate(tqdm(image_ids)):
		embedding = embed_model.encode([image_metadata['Caption'][i]], convert_to_tensor=True)
		embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
		embedding = embedding[:, :matryoshka_dim]
		embedding = F.normalize(embedding, p=2, dim=1)
		caption_embeddings.append(embedding)

	caption_embeddings = torch.cat(caption_embeddings, dim=0)

	print('Time to encode video descriptions: ', (time.time() - start_time), 's')

@app.get('/')
async def upload_form(request: Request):
	global image_ids

	image_count = len(image_ids)
	formatted_image_count = "{:,}".format(image_count)

	return templates.TemplateResponse('autocomplete.html', {"request": request, "image_count": formatted_image_count})

@app.get("/search")
async def search(term: str):

	global embed_model
	global image_ids
	global caption_embeddings
	global language_model
	global tokenizer
	global image_metadata

	print('Search term for autocompletion: ', term)

	# Compute time to autocomplete
	start_time = time.time()

	# Get top-10 similar image ids
	similar_image_ids = get_image_ids_from_search_term(
		embed_model,
		term, 
		image_ids, 
		caption_embeddings, 
		vector_dim=matryoshka_dim, 
		threshold=similarity_threshold, 
		top_k=top_autocomplete_images
	)

	# If there are less than 5 similar images, use all of them
	len_random_image = 5 if len(similar_image_ids) >= 5 else len(similar_image_ids)

	# Randomly select images id
	selected_image_ids = np.random.choice(similar_image_ids, len_random_image, replace=False)

	# Get caption and object detection result
	captions = []
	ods = []
	for selected_image_id in selected_image_ids:
		caption = image_metadata[image_metadata['ImageID'] == selected_image_id]['Caption'].values[0]
		od = image_metadata[image_metadata['ImageID'] == selected_image_id]['OD'].values[0]
		captions.append(caption)
		ods.append(od)

	# Make input message for autocompletion
	input_message = init_msg + '\n\n' + '--' + '\n'
	for i in range(len_random_image):
		input_message += f'Scene #{i+1}' + '\n'
		input_message += '<DETAILED_CAPTION>' + '\n' + captions[i] + '\n' + '</DETAILED_CAPTION>' + '\n'
		input_message += '<OD>' + '\n' + ods[i] + '\n' + '</OD>' + '\n'
		input_message += '<SEARCH_PHRASE>' + term + '</SEARCH_PHRASE>' + '\n'
		input_message += '--' + '\n'
	input_message += 'Highlight a completed search phrases using tag <COMPLETED></COMPLETED>' + '\n'
	input_message += "If the search phrase contains completed information, add other information from caption to make it more detailed using 'and' notation." + '\n'
	input_message += "Make autocompletion from above 5 scenarios." + '\n'
	input_message += "Only give me the completed search phrase." + '\n'
	input_message += "You should make autocomplete results even if the search phrases are longer than 10 words." + '\n'
	input_message += "Must avoid generating the same autocomplete results."

	# Generate autocompletion
	messages = [
		{"role": "user", "content": input_message},
	]

	# Apply chat template to input messages
	input_ids = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		return_tensors="pt"
	).to(language_model.device)

	# Terminators for autocompletion
	terminators = [
		tokenizer.eos_token_id,
		tokenizer.convert_tokens_to_ids("<|eot_id|>")
	]

	# Generate autocompletion
	outputs = language_model.generate(
		input_ids,
		max_new_tokens=max_new_tokens,
		eos_token_id=terminators,
		do_sample=True,
		temperature=0.6,
		top_p=0.9,
	)

	# Decode autocompletion results
	response = outputs[0][input_ids.shape[-1]:]
	decoded_response = tokenizer.decode(response, skip_special_tokens=True)
	# print('Autocomplete results: ', decoded_response)

	# Get autocompletion results extracting text between <COMPLETED> and </COMPLETED>
	# Ignore if there is no close tag </COMPLETED>
	autocomplete_results = [item.strip() for item in decoded_response.split('<COMPLETED>')[1:]]
	autocomplete_results = [item.split('</COMPLETED>')[0] if '</COMPLETED>' in item else '' for item in autocomplete_results]

	# Remove empty strings
	autocomplete_results = [item for item in autocomplete_results if item != '']

	print('Time to autocomplete: ', (time.time() - start_time), 's')

	return autocomplete_results

# @app.post("/get_list")
@app.get("/get_list")
async def get_list(text: str):

	global embed_model
	global image_ids
	global caption_embeddings

	print('Search term for image search: ', text)

	if text != '':
		start_time = time.time()

		# Get top-k image ids
		similar_image_ids = get_image_ids_from_search_term(
			embed_model,
			text, 
			image_ids, 
			caption_embeddings, 
			vector_dim=matryoshka_dim, 
			threshold=similarity_threshold, 
			top_k=top_similar_images
		)

		print('Time to search images: ', (time.time() - start_time), 's')

		# Return top-k image paths
		# return [os.path.join('static', images_base, item + '.png') for item in similar_image_ids[:top_similar_images]]
		return [os.path.join('static', images_base, item + '.jpg') for item in similar_image_ids[:top_similar_images]]

	else:
		return []

@app.get("/log_key_event")
async def log_key_event(request: Request, text: str):

	client_ip = request.client.host
	current_time = datetime.now().isoformat()

	print(f'Client IP: {client_ip}, Text: {text}, Time: {current_time}')
	
	return {"message": "Log saved successfully"}


if __name__ == "__main__":
    init()
    # uvicorn.run(app, port=8080)
    uvicorn.run(app, host='0.0.0.0', port=8080)