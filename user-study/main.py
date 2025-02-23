import os, json
import duckdb
import pandas as pd
import time
import ast
import emoji
from pathlib import Path
import numpy as np

import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from sentence_transformers import SentenceTransformer

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Set up template and static directories
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=templates_dir)
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Path to the phrase-emoji CSV file
phrase_emoji_path = Path(__file__).parent.parent / 'outputs' / 'drama-vid-1k-wRisk-pllava13b-llama3_1_70b-phrase-one-emoji-no-atmo-w-ex.csv'

# Declare global variables
model = None
video_embeddings = {}  # key: vid_id, value: embedding vector

def init():
    global model, video_embeddings
    df = pd.read_csv(phrase_emoji_path)
    phrase_emoji_df = pd.DataFrame(columns=['vidID', 'phrase', 'emoji'])

    print("Loading phrase-emoji data...")
    for index, row in df.iterrows():
        vidid = row['Video ID']
        phrase_emoji = row['PhraseEmoji']
        try:
            phrase_emoji = ast.literal_eval(phrase_emoji)
        except SyntaxError:
            emojis_list = emoji.emoji_list(phrase_emoji)
            emojis_only_list = [e['emoji'] for e in emojis_list]
            phrase_only_string = phrase_emoji
            for e in emojis_list:
                phrase_only_string = phrase_only_string.replace(e['emoji'], '""')
            try:
                phrase_only_list = ast.literal_eval(phrase_only_string)
            except SyntaxError:
                continue
            for i, e in enumerate(emojis_only_list):
                phrase_only_list[i]['emoji'] = e
            phrase_emoji = phrase_only_list
        for item in phrase_emoji:
            phrase_emoji_df = pd.concat([phrase_emoji_df, 
                pd.DataFrame([[vidid, item['phrase'], item['emoji']]], columns=['vidID', 'phrase', 'emoji'])])
    phrase_emoji_df.reset_index(drop=True, inplace=True)
    duckdb.sql('CREATE TABLE drama AS SELECT * FROM phrase_emoji_df')

    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("Loading embeddings...")
    # Load video description embeddings from the embeddings directory
    embeddings_dir = Path(__file__).parent.joinpath('embeddings')
    video_embeddings = {}
    for file in embeddings_dir.iterdir():
        if file.suffix == '.npy':
            vid_id = file.stem
            video_embeddings[vid_id] = np.load(file)
    print("Initialization complete.")

# Login page
@app.get("/")
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Login validation and redirect to demo page
@app.post("/login")
async def login(request: Request, user_id: str = Form(...)):
    allowed_ids = {"12345", "67890", "54321", "09876", "11223"}
    if user_id in allowed_ids:
        request.session["user_id"] = user_id
        return RedirectResponse(url="/demo", status_code=302)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid ID."})

# Demo page (demo.html)
@app.get("/demo")
async def demo_page(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("demo.html", {"request": request})

# Actual test page (emoji_autocomplete.html)
@app.get("/emoji_autocomplete")
async def test_page(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("emoji_autocomplete.html", {"request": request})

# Text-only autocompletion endpoint
@app.get("/search_text")
async def search_text(term: str):
    res = duckdb.sql("SELECT phrase FROM drama WHERE phrase ILIKE '%" + term + "%'").df()
    results = [{'label': row['phrase']} for index, row in res.iterrows()]
    return results[:20]

# Text-emoji autocompletion endpoint
@app.get("/search_emoji")
async def search_emoji(term: str):
    res = duckdb.sql("SELECT phrase, emoji FROM drama WHERE phrase ILIKE '%" + term + "%'").df()
    results = [{'label': row['phrase'], 'emoji': row['emoji']} for index, row in res.iterrows()]
    return results[:20]

# get_list endpoint: returns the top 30 related videos (video URL and thumbnail URL) based on cosine similarity
@app.post("/get_list")
async def get_list(searchTxt: str = Form(...)):
    global model, video_embeddings
    query = searchTxt.strip()
    if not query:
        return []
    # Compute the embedding for the query text
    query_emb = model.encode(query)
    scores = []
    # Calculate cosine similarity between query embedding and each video embedding
    for vid_id, emb in video_embeddings.items():
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
        scores.append((vid_id, sim))
    # Sort scores in descending order and select top 30
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top30 = scores[:30]
    results = []
    for vid_id, score in top30:
        video_url = "static/drama-1k-vids/" + vid_id + ".mp4"
        thumbnail_url = "static/drama-1k-imgs/" + vid_id + ".jpg"
        results.append({"video_url": video_url, "thumbnail_url": thumbnail_url})
    return results

# Common search endpoint (if needed)
@app.get("/search")
async def search(request: Request, term: str):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    res = duckdb.sql("SELECT * FROM drama WHERE phrase ILIKE '%" + term + "%'").df()
    res = res.sort_values(by='phrase', key=lambda col: col.str.startswith(term), ascending=False)
    results = [{'label': row['phrase'], 'emoji': row['emoji']} for index, row in res.iterrows()]
    return results[:20]

if __name__ == "__main__":
    init()
    uvicorn.run(app, host="0.0.0.0")
