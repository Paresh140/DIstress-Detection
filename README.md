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
| `python -m app.cv` | Leave-speakers-out cross-validation: six folds over all 24 RAVDESS actors, or `--corpus crema-d` for five folds over the 91 CREMA-D speakers. Runs the CNN, an eGeMAPS + logistic regression baseline, and a frozen WavLM-base-plus encoder with attentive pooling and a four-class head (neutral or calm, sad, angry, fearful). Each fold is scored raw (threshold 0.5) and calibrated (temperature and threshold fitted on validation speakers), with ECE, and models are compared with paired tests over folds. `--only cnn\|egemaps\|wavlm` runs one model; `--head binary-mean` gives the first WavLM head; `--encoder facebook/wav2vec2-base` swaps the encoder. | `artifacts/cv_report.json`, `artifacts/cv_report_crema-d.json`, `artifacts/ssl_cache/` |
| `python -m app.study` | Cross-corpus study, RAVDESS ↔ CREMA-D (needs `data/crema-d/`), for the CNN, eGeMAPS and WavLM on the same speaker splits. `--models wavlm egemaps` picks models; `--encoder` and `--head` work as in `app.cv`; `--norm corpus\|speaker` standardizes features per corpus or speaker with unlabeled audio. Every run is also calibrated, with a speaker-bootstrap 95% interval. | `artifacts/cross_corpus_report.json` |
| `python -m app.compare` | Reads the three reports above and writes one comparison table plus a check against the proposal targets. | `artifacts/comparison.md` |

Labels everywhere: distress = sad, angry, fearful; non-distress = neutral, calm; happy, disgust and surprise are excluded.

The WavLM model downloads its weights from Hugging Face the first time it runs (about 380 MB; on Kaggle, turn Internet on). Every clip is embedded once and cached, so later runs only train the head.

Checkpoints and hyperparameters are chosen on validation speakers by balanced accuracy, never on test speakers. Reports include balanced accuracy, macro F1, AUROC, specificity and the F1 of always predicting distress, which is the baseline to beat.
