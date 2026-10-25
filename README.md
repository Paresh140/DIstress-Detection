# DIstress-Detection

DistressNet: a voice-based distress detector (log-mel CNN on the voice, an LLM reading the words, a rule-based judge) with a supportive chat on top. Not a clinical tool.

## Setup

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# Frontend (second terminal)
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api to :8000
```

Put RAVDESS speech (`Audio_Speech_Actors_01-24.zip` from Zenodo record 1188976) under `data/ravdess/Actor_01 … Actor_24/`. Set `OPENAI_API_KEY` to use gpt-4o-mini for the words; otherwise a local Ollama model is used.

## Training and evaluation

All commands run from `backend/`.

| Command | What it does | Output |
|---|---|---|
| `python -m app.train` | Trains the Monitor model. Actors 01–18 train, 19–20 choose the epoch, 21–24 are the test set (same split as the Colab run). | `artifacts/audio_branch.pt`, `artifacts/training_report.json` |
| `python -m app.cv` | Six-fold leave-speakers-out cross-validation over all 24 actors, for the CNN, an eGeMAPS + logistic regression baseline, and a frozen WavLM-base-plus encoder with a small trained head. Mean ± std per metric. `--only cnn\|egemaps\|wavlm` runs one model; `--encoder facebook/wav2vec2-base` swaps the pretrained encoder. | `artifacts/cv_report.json`, `artifacts/ssl_cache/` |
| `python -m app.study` | Cross-corpus study, RAVDESS ↔ CREMA-D (needs `data/crema-d/`). | `artifacts/cross_corpus_report.json` |

Labels everywhere: distress = sad, angry, fearful; non-distress = neutral, calm; happy, disgust and surprise are excluded.

The WavLM model downloads its weights from Hugging Face the first time it runs (about 380 MB; on Kaggle, turn Internet on). Every clip is embedded once and cached, so later runs only train the head.

Checkpoints and hyperparameters are chosen on validation speakers by balanced accuracy, never on test speakers. Reports include balanced accuracy, macro F1, AUROC, specificity and the F1 of always predicting distress, which is the baseline to beat.
