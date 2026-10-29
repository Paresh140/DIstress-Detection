# DIstress-Detection

DistressNet: a voice-based distress detector with a supportive chat on top. Not a clinical tool.

How one clip is read:

1. **Voice.** A frozen WavLM-base-plus encoder, a learned mix of its 13 layers, attentive statistics pooling and a four-class head (non-distress, sad, angry, fearful). P(distress) is the sum of the three negative classes, temperature-scaled on validation speakers. Until `artifacts/voice_head.pt` exists, the older log-mel CNN is used.
2. **Words.** faster-whisper transcribes the clip (or the browser's or typed transcript is used), and a language model scores the words, with a phrase-list fallback.
3. **Crisis screen.** A keyword floor for explicit risk plus a language-model risk label (none, concern, high) on every message. High risk skips everything else and gives the 988 referral.
4. **Judge.** Each side is put in a band: calm, check-in or support. The voice is banded around its validated threshold. The judge keeps the higher band, so neither side can overrule distress from the other.
5. **Support agent.** On check-in or support, a short conversation with state (turn, risk, step, what was shared): reflect, ask what would help, offer one grounding idea or resource, check in, keep listening. It says it is not a counselor and never shows scores; those stay in the studio views.

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

Optional: `pip install faster-whisper` for speech to text (`WHISPER_MODEL=small` for better accuracy).

Put RAVDESS speech (`Audio_Speech_Actors_01-24.zip` from Zenodo record 1188976) under `data/ravdess/Actor_01 … Actor_24/`. Set `OPENAI_API_KEY` to use gpt-4o-mini for the words; otherwise a local Ollama model is used.

## Training and evaluation

All commands run from `backend/`.

| Command | What it does | Output |
|---|---|---|
| `python -m app.train` | Trains the Monitor model. Actors 01–18 train, 19–20 choose the epoch, 21–24 are the test set (same split as the Colab run). | `artifacts/audio_branch.pt`, `artifacts/training_report.json` |
| `python -m app.cv` | Leave-speakers-out cross-validation: six folds over all 24 RAVDESS actors, or `--corpus crema-d` for five folds over the 91 CREMA-D speakers. Runs the CNN, an eGeMAPS + logistic regression baseline, and a frozen WavLM-base-plus encoder with attentive pooling and a four-class head (neutral or calm, sad, angry, fearful). Each fold is scored raw (threshold 0.5) and calibrated (temperature and threshold fitted on validation speakers), with ECE, and models are compared with paired tests over folds. `--only cnn\|egemaps\|wavlm` runs one model; `--head binary-mean` gives the first WavLM head; `--encoder facebook/wav2vec2-base` swaps the encoder. | `artifacts/cv_report.json`, `artifacts/cv_report_crema-d.json`, `artifacts/ssl_cache/` |
| `python -m app.study` | Cross-corpus study, RAVDESS ↔ CREMA-D (needs `data/crema-d/`), for the CNN, eGeMAPS and WavLM on the same speaker splits. `--models wavlm egemaps` picks models; `--encoder` and `--head` work as in `app.cv`; `--norm corpus\|speaker` standardizes features per corpus or speaker with unlabeled audio. Every run is also calibrated, with a speaker-bootstrap 95% interval. | `artifacts/cross_corpus_report.json` |
| `python -m app.voice` | Trains the live WavLM voice model: RAVDESS actors 01–18 plus CREMA-D (when present) train, actors 19–20 and nine CREMA-D speakers choose the epoch, temperature and threshold, actors 21–24 are scored once. A second head is trained on features shifted by each speaker's calm enrollment (two neutral clips, left out of every split). Copy the file into `artifacts/` on the studio machine and Monitor uses it; after you save two calm readings on Monitor, it switches to the enrolled head. | `artifacts/voice_head.pt` |
| `python -m app.safety_eval` | Runs the crisis screen and the agent on the 100-message safety set (`eval/safety_set.jsonl`) and checks the targets: every explicit message gets 988, at least 90% of indirect messages are flagged, at most 10% of harmless ones are flagged high. Exits 1 on a miss. `--no-model` tests the keyword floor alone. | `artifacts/safety_report.json` |
| `python -m app.reader_eval` | Compares the phrase list, the language-model reader and a GoEmotions RoBERTa on 200 labeled sentences (`eval/reader_sentences.jsonl`), overall and per group. | `artifacts/reader_report.json` |
| `python -m pytest tests` | Unit tests for the judge bands and floor, the crisis screen, the support flow and calibration. No model calls. | |
| `python -m app.compare` | Reads the three reports above and writes one comparison table plus a check against the proposal targets. | `artifacts/comparison.md` |

Labels everywhere: distress = sad, angry, fearful; non-distress = neutral, calm; happy, disgust and surprise are excluded.

The WavLM model downloads its weights from Hugging Face the first time it runs (about 380 MB; on Kaggle, turn Internet on). Every clip is embedded once and cached, so later runs only train the head.

Checkpoints and hyperparameters are chosen on validation speakers by balanced accuracy, never on test speakers. Reports include balanced accuracy, macro F1, AUROC, specificity and the F1 of always predicting distress, which is the baseline to beat.
