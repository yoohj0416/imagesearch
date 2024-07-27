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
templates_dir = "/home/hojin/imagesearch/query/templates"
templates = Jinja2Templates(directory=templates_dir)
# templates = Jinja2Templates(directory="templates")
static_dir = "/home/hojin/imagesearch/query/static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/static", StaticFiles(directory="/pool0/data_archive_hojin/drama_data"), name="static")

# Hyperparameters for embedding model
embed_model_id = "nomic-ai/nomic-embed-text-v1.5"
matryoshka_dim = 512
similarity_threshold = 0.55
top_similar_images = 300

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
	with open(log_file_name, 'w') as f:
		f.write('')

global embed_model
global image_ids
global caption_embeddings
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
	global image_metadata

	# Load embedding model for similarity search
	embed_model = SentenceTransformer(embed_model_id, trust_remote_code=True)

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

	return templates.TemplateResponse('no_autocomplete.html', {"request": request, "image_count": formatted_image_count})

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
    # uvicorn.run(app, host='0.0.0.0', port=8080)
    uvicorn.run(app, host='0.0.0.0', port=80)