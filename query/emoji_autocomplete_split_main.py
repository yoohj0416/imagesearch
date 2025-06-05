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

# Use the new CSV file with split phrases, emojis, and importance scores
# phrase_emoji_path = Path(__file__).parent.parent / 'outputs' / 'drama-vid-1k-wRisk-pllava13b-llama3_3-70b-phrases-emojis-split-importance.csv'
phrase_emoji_path = Path(__file__).parent.parent / 'outputs' / 'msvd-pllava13b-llama3_3-70b-phrases-emojis-split-importance.csv'

# CSV file for video descriptions
description_path = Path(__file__).parent.parent / 'outputs' / 'msvd-pllava13b-descriptions-no-atmo.csv'

# vid_base = 'drama-video-100-mp4'
vid_base = 'msvd-vids'
thumbnail_base = 'msvd-imgs'

# Global variables
model = None
video_embeddings = {}  # key: vid_id, value: embedding vector
descriptions = {}  # key: vid_id, value: description text

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df_out = pd.DataFrame(columns=['vidID', 'phrase', 'split', 'emojis', 'importance'])
    for index, row in df.iterrows():
        vidid = row['Video ID']
        phrase_emoji = row['PhraseEmojis']
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
            if any(e.strip() == "" for e in item['emojis']):
                continue
            new_row = pd.DataFrame([[vidid, item['phrase'], json.dumps(item['split']),
                                      json.dumps(item['emojis']), json.dumps(item['importance'])]],
                                     columns=['vidID', 'phrase', 'split', 'emojis', 'importance'])
            df_out = pd.concat([df_out, new_row], ignore_index=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out

def load_embeddings(embeddings_folder):
    emb_dict = {}
    embeddings_dir = Path(__file__).parent.joinpath(embeddings_folder)
    for file in embeddings_dir.iterdir():
        if file.suffix == '.npy':
            vid_id = file.stem
            emb_dict[vid_id] = np.load(file)
    return emb_dict

def load_descriptions(desc_csv_path):
    desc_dict = {}
    df = pd.read_csv(desc_csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    for index, row in df.iterrows():
        vidid = row["Video ID"]
        answer = row["Answer"]
        desc_dict[vidid] = answer
    return desc_dict

def init():
    global model, video_embeddings, descriptions

    print("Initializing...")
    # Load the dataset
    phrase_emoji_df = load_dataset(phrase_emoji_path)
    print(f"Loaded {len(phrase_emoji_df)} rows from {phrase_emoji_path}")
    duckdb.register('phrase_emoji_df', phrase_emoji_df)
    duckdb.sql('CREATE TABLE drama AS SELECT * FROM phrase_emoji_df')

    # Load video descriptions
    descriptions = load_descriptions(description_path)
    print(f"Loaded {len(descriptions)} video descriptions from {description_path}")

    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("Loading video embeddings...")
    video_embeddings = load_embeddings("embeddings/msvd")

    print("Initialization complete.")

    # print("Loading embeddings...")
    # embeddings_dir = Path(__file__).parent.joinpath('embeddings')
    # video_embeddings = {}
    # for file in embeddings_dir.iterdir():
    #     if file.suffix == '.npy':
    #         vid_id = file.stem
    #         video_embeddings[vid_id] = np.load(file)
    # print("Initialization complete.")

@app.get("/")
async def upload_form(request: Request):
    return templates.TemplateResponse("emoji_autocomplete.html", {"request": request})

@app.get("/search")
async def search(term: str):
    start_time = time.time()
    print('term: ', term)
    res = duckdb.sql("SELECT * FROM drama WHERE phrase ILIKE '%" + term + "%'").df()
    res = res.sort_values(by='phrase', key=lambda col: col.str.startswith(term), ascending=False)
    results = [{'label': row['phrase'], 'emoji': row['emojis']} for index, row in res.iterrows()]
    resp = results[:20]
    print(f"Time to autocomplete: {(time.time() - start_time) * 1000} ms")
    return resp

@app.get("/search_emoji")
async def search_emoji(term: str):
    # New emoji autocompletion logic using split, emojis, and importance
    res = duckdb.sql("SELECT * FROM drama WHERE phrase ILIKE '%" + term + "%'").df()
    suggestions = []
    for index, row in res.iterrows():
        try:
            split_list = json.loads(row['split'])
            emoji_list = json.loads(row['emojis'])
            importance_list = json.loads(row['importance'])
        except Exception as e:
            continue
        suggestion = None
        # For each segment, if the user's term is found, determine which emoji to suggest.
        for i, segment in enumerate(split_list):
            if term.lower() in segment.lower():
                ratio = len(term) / len(segment)
                if ratio < 0.6:
                    suggestion = emoji_list[i]
                    break
                else:
                    # If user input is 80% or more of the segment, choose an alternative emoji
                    # among the remaining ones based on the highest importance score.
                    candidates = []
                    for j in range(len(importance_list)):
                        if j != i:
                            candidates.append((j, importance_list[j]))
                    if len(candidates) > 0:
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        suggestion = emoji_list[candidates[0][0]]
                    else:
                        suggestion = emoji_list[i]
                    break
        if suggestion:
            suggestions.append({'label': row['phrase'], 'emoji': suggestion})
    return suggestions[:20]

@app.post("/get_list")
async def get_list(searchTxt: str = Form(...)):
    global model, video_embeddings, descriptions
    query = searchTxt.strip()
    if not query:
        return []
    query_emb = model.encode(query)
    scores = []
    for vid_id, emb in video_embeddings.items():
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
        scores.append((vid_id, sim))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top30 = scores[:30]
    results = []
    for vid_id, score in top30:
        video_url="static/" + vid_base + "/" + vid_id + ".mp4"
        thumbnail_url = "static/" + thumbnail_base + "/" + vid_id + ".jpg"
        description = descriptions.get(vid_id, "No description available.")
        if len(description) > 200:
            description = description[:200] + "..."
        results.append({"video_url": video_url, "thumbnail_url": thumbnail_url, "description": description})
    return results

@app.get("/search")
async def common_search(request: Request, term: str):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    res = duckdb.sql("SELECT * FROM drama WHERE phrase ILIKE '%" + term + "%'").df()
    res = res.sort_values(by='phrase', key=lambda col: col.str.startswith(term), ascending=False)
    results = [{'label': row['phrase'], 'emoji': row['emojis']} for index, row in res.iterrows()]
    return results[:20]

if __name__ == "__main__":
    init()
    uvicorn.run(app, host="0.0.0.0")
