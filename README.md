# MoodTune — fresh starter

This folder starts a new local Python prototype for the agreed flow:

1. Type how you feel, in English.
2. The model predicts one emotion.
3. Choose Malayalam, English, or Hindi for the songs.
4. Show five matching songs.

## Model and dataset

`train_model.py` trains **TF-IDF + Logistic Regression** on the real English `dair-ai/emotion` dataset. It uses that dataset's train and validation splits for learning and its held-out test split for the accuracy, precision, recall, F1 report, and confusion matrix. Dataset details: https://huggingface.co/datasets/dair-ai/emotion

The dataset has six labels: sadness, joy, love, anger, fear, and surprise. It has no neutral label. This is our own trained scikit-learn classifier; it does not use the previously downloaded AnasAlokla model.

## Windows / VS Code

Open this folder in VS Code, then open **Terminal → New Terminal** and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

Training downloads the dataset the first time, so that run needs internet. It prints the actual metrics and saves `artifacts\vectorizer.joblib` and `artifacts\classifier.joblib`.

Then try the recommendation flow:

```powershell
python recommend.py
```

## Song recommendations

`recommend.py` and `api.py` read the project-local `MoodTunes_Expanded_Final.csv`, so the catalogue stays with the code when the project is copied or deployed. Its columns are `song_name`, `artist_name`, `language`, and `mood`. It has 20–22 songs in each of the 15 language/mood groups, so it can return five songs per request with more variety.

The classifier emotions map to the catalogue moods as follows:

- joy → happy; surprise → energetic/motivated; love → romantic
- sadness → sad; anger and fear → calm/focussed

The TF-IDF preprocessor marks words in a short negation scope (for example,
`not happy`) as separate features. The app also has a narrow fallback for an
explicitly negated positive phrase when no other emotion cue is present: it
chooses sadness, the closest available negative-valence class because this
dataset has no neutral label. This is a rule-based fallback, not a learned
classifier result; reported test metrics measure the classifier alone.

At the language prompt, enter Malayalam, English, or Hindi. Short forms `mal`, `eng`, and `hind` also work. The catalogue has no listening-link column, so the prototype displays song titles and artists only.

## API for the frontend

The API uses the same trained model and song catalogue. In the activated `.venv`, update/install dependencies and start the local server:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The frontend can send a `POST` request to `http://127.0.0.1:8000/recommend` with JSON such as:

```json
{"text": "I am feeling sad", "language": "English"}
```

The response contains the predicted emotion, matched music mood, selected language, and five songs. Open `http://127.0.0.1:8000/docs` to see and try the API in a browser. This address works on the same computer; a frontend running on another computer will need the backend hosted or exposed on a shared network.

## Publish the shareable Streamlit app

`app.py` runs the saved TF-IDF + Logistic Regression model directly and reads the bundled song CSV. It does not need the local FastAPI server. Keep `app.py`, `requirements.txt`, `MoodTunes_Expanded_Final.csv`, and the complete `artifacts` folder together when publishing. Do not upload `.venv`.

To share the app online, put this folder's project files in a GitHub repository, then deploy `app.py` from that repository using [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). After deployment, share the resulting `streamlit.app` link with your friend; they can open it from their computer without running Python locally.
