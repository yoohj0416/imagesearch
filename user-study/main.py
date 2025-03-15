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

# Use the correct CSV files for both datasets
drama_csv_path = Path(__file__).parent.parent / 'outputs' / 'drama-vid-1k-wRisk-pllava13b-llama3_3-70b-phrases-emojis-split-importance.csv'
msvd_csv_path = Path(__file__).parent.parent / 'outputs' / 'msvd-pllava13b-llama3_3-70b-phrases-emojis-split-importance.csv'

drama_vid_base = 'drama-video-100-mp4'
# For MSVD, videos are in "msvd-vids" and thumbnails in "msvd-imgs"

# Global variables
model = None
# video_embeddings is a dictionary with keys "drama" and "msvd"
video_embeddings = {"drama": {}, "msvd": {}}

def load_dataset(csv_path):
    # Read CSV using utf-8-sig and strip column names
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

def init():
    global model, video_embeddings
    # Load drama dataset and create DuckDB table
    drama_df = load_dataset(drama_csv_path)
    duckdb.sql('CREATE TABLE drama AS SELECT * FROM drama_df')
    # Load msvd dataset and create DuckDB table
    msvd_df = load_dataset(msvd_csv_path)
    duckdb.sql('CREATE TABLE msvd AS SELECT * FROM msvd_df')
    
    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Loading drama embeddings...")
    video_embeddings["drama"] = load_embeddings("embeddings/drama-1k")
    print("Loading msvd embeddings...")
    video_embeddings["msvd"] = load_embeddings("embeddings/msvd")
    print("Initialization complete.")

@app.get("/")
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, user_id: str = Form(...)):
    # Define groups: text_only and emoji_text
    group_text_only = {"12345", "54321"}
    group_emoji_text = {"67890", "09876", "11223"}
    if user_id in group_text_only or user_id in group_emoji_text:
        request.session["user_id"] = user_id
        if user_id in group_text_only:
            request.session["test_group"] = "text_only"
        else:
            request.session["test_group"] = "emoji_text"
        return RedirectResponse(url="/demo", status_code=302)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid ID."})

@app.get("/demo")
async def demo_page(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("demo.html", {"request": request, "test_group": request.session.get("test_group")})

@app.get("/emoji_autocomplete")
async def test_page(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("emoji_autocomplete.html", {"request": request, "test_group": request.session.get("test_group")})

@app.get("/search_text")
async def search_text(term: str, dataset: str = "drama"):
    table = dataset  # either 'drama' or 'msvd'
    res = duckdb.sql("SELECT phrase FROM " + table + " WHERE phrase ILIKE '%" + term + "%'").df()
    results = [{'label': row['phrase']} for index, row in res.iterrows()]
    return results[:20]

@app.get("/search_emoji")
async def search_emoji(term: str, dataset: str = "drama"):
    table = dataset
    res = duckdb.sql("SELECT * FROM " + table + " WHERE phrase ILIKE '%" + term + "%'").df()
    suggestions = []
    for index, row in res.iterrows():
        try:
            split_list = json.loads(row['split'])
            emoji_list = json.loads(row['emojis'])
            importance_list = json.loads(row['importance'])
        except Exception as e:
            continue
        suggestion = None
        for i, segment in enumerate(split_list):
            if term.lower() in segment.lower():
                ratio = len(term) / len(segment)
                if ratio < 0.6:
                    suggestion = emoji_list[i]
                    break
                else:
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
async def get_list(searchTxt: str = Form(...), dataset: str = Form("drama")):
    global model, video_embeddings
    query = searchTxt.strip()
    if not query:
        return []
    query_emb = model.encode(query)
    scores = []
    for vid_id, emb in video_embeddings.get(dataset, {}).items():
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
        scores.append((vid_id, sim))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top30 = scores[:30]
    results = []
    for vid_id, score in top30:
        if dataset == "msvd":
            # video_url = "static/msvd-vids/" + vid_id + ".avi"
            video_url = "static/msvd-vids/" + vid_id + ".mp4"
            thumbnail_url = "static/msvd-imgs/" + vid_id + ".jpg"
        else:
            video_url = "static/drama-1k-vids/" + vid_id + ".mp4"
            thumbnail_url = "static/drama-1k-imgs/" + vid_id + ".jpg"
        results.append({"video_url": video_url, "thumbnail_url": thumbnail_url})
    return results

@app.get("/search")
async def common_search(request: Request, term: str, dataset: str = "drama"):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    res = duckdb.sql("SELECT * FROM " + dataset + " WHERE phrase ILIKE '%" + term + "%'").df()
    res = res.sort_values(by='phrase', key=lambda col: col.str.startswith(term), ascending=False)
    results = [{'label': row['phrase'], 'emoji': row['emojis']} for index, row in res.iterrows()]
    return results[:20]

@app.post("/log_event")
async def log_event(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return {"status": "error", "message": "Not logged in"}
    data = await request.json()
    logs_dir = Path(__file__).parent.joinpath("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"{user_id}.csv"
    with open(log_file, "a", encoding="utf-8") as f:
        line = f'{data.get("timestamp")},{data.get("event_type")},{data.get("stage")},{data.get("topic")},{data.get("details")}\n'
        f.write(line)
    return {"status": "ok"}

if __name__ == "__main__":
    init()
    uvicorn.run(app, host="0.0.0.0", port=8080)
