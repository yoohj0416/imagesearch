import os, json
import duckdb
import pandas as pd
import time
import ast
import emoji
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles


app = FastAPI()
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=templates_dir)
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

phrase_emoji_path = Path(__file__).parent.parent / 'outputs' / 'drama-vid-1k-wRisk-pllava13b-llama3_1_70b-phrase-one-emoji-no-atmo-w-ex.csv'

drama_vid_base = 'drama-video-100-mp4'

def init():

	# global phrase_emoji_df

	# Load the data using pandas
	df = pd.read_csv(phrase_emoji_path)

	# Preprocess the data
	# Iterate through the dataframe and make new df whose columns are vidID, phrase, emoji
	phrase_emoji_df = pd.DataFrame(columns=['vidID', 'phrase', 'emoji'])

	for index, row in df.iterrows():
		vidid = row['Video ID']
		phrase_emoji = row['PhraseEmoji']
		try:
			phrase_emoji = ast.literal_eval(phrase_emoji)
		except SyntaxError:
			emojis_list = emoji.emoji_list(phrase_emoji)
			# Add ' in before 'match_start' and after 'match_end' in the phrase_emoji
			emojis_only_list = [emojis['emoji'] for emojis in emojis_list]
			phrase_only_string = phrase_emoji
			for emojis in emojis_list:
				phrase_only_string = phrase_only_string.replace(emojis['emoji'], '""')

			try:
				phrase_only_list = ast.literal_eval(phrase_only_string)
			except SyntaxError:
				continue

			for i, emojis in enumerate(emojis_only_list):
				phrase_only_list[i]['emoji'] = emojis

			phrase_emoji = phrase_only_list

		for item in phrase_emoji:
			phrase_emoji_df = pd.concat([phrase_emoji_df, pd.DataFrame([[vidid, item['phrase'], item['emoji']]], columns=['vidID', 'phrase', 'emoji'])])

	# Reindex the dataframe
	phrase_emoji_df.reset_index(drop=True, inplace=True)

	# Create table in duckdb
	duckdb.sql('CREATE TABLE drama AS SELECT * FROM phrase_emoji_df')

@app.get("/")
async def upload_form(request: Request):
    return templates.TemplateResponse("emoji_autocomplete.html", {"request": request})

@app.get("/search")
async def search(term: str):

	# compute time to autocomplete
	start_time = time.time()

	print ('term: ', term)

	res = duckdb.sql('SELECT * FROM drama WHERE phrase ILIKE \'%' + term + '%\'').df()

	# Sort the result by the phrase that starts with the term
	res = res.sort_values(by='phrase', key=lambda col: col.str.startswith(term), ascending=False) 

	# return the first 20 phrases and emojis
	results = [{'label': row['phrase'], 'emoji': row['emoji']} for index, row in res.iterrows()]
	resp = results[:20]

	# Print the time to autocomplete in milliseconds
	print(f"Time to autocomplete: {(time.time() - start_time) * 1000} ms")
	
	return resp


# @app.route('/get_list', methods=['POST'])
# def get_list():

# 	text = request.form.get('searchTxt')
# 	print(f'keyword: {text}')

# 	conn = duckdb.connect(":default:")
# 	query = f"""
# 	SELECT *
# 	FROM drama
# 	WHERE keyword1 ILIKE '%{text}%' OR keyword2 ILIKE '%{text}%' OR keyword3 ILIKE '%ILIKE{text}%'
# 	"""
# 	df = conn.sql(query).df()

# 	filtered_dict = df['Image'].tolist()
# 	# remove redundant items
# 	filtered_dict = list(set(filtered_dict))

# 	# combine the image name with the drama_vid_base and suffix '.mp4'
# 	filtered_dict = [os.path.join(drama_vid_base, item + '.mp4') for item in filtered_dict]

# 	# make path to url with url_for
# 	video_urls = [url_for('static', filename=item) for item in filtered_dict]

# 	# resp = jsonify(filtered_dict)
# 	resp = jsonify(video_urls)

# 	resp.status_code = 200

# 	return resp


if __name__ == "__main__":
	init()
	uvicorn.run(app, host="0.0.0.0", port=8000)