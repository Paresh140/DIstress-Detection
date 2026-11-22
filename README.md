# DIstress-Detection

DistressNet: a voice-based distress detector with a supportive chat on top. Not a clinical tool.

How one clip is read:

1. **Voice.** A frozen WavLM-base-plus encoder, a learned mix of its 13 layers, mean pooling over time and a binary head (`binary-mean`, the live head: best in speaker cross-validation and across corpora). P(distress) is temperature-scaled on validation speakers. The proposal's four-class attentive head is kept only as a reported ablation (`--head four-attentive`). When the person has calibrated (below), every clip is z-scored against their own calm reading and scored by the calm-enrolled head. Until `artifacts/voice_head.pt` exists, the older log-mel CNN is used.
2. **Words.** faster-whisper transcribes the clip (or the browser's or typed transcript is used), and a language model scores the words. If no model answers, the phrase list is the live fallback. The GoEmotions RoBERTa reader is an offline comparison in `app.reader_eval` only; it is not used live.
3. **Crisis screen.** A keyword floor for explicit risk plus a language-model risk label (none, concern, high) on every message. High risk skips everything else and gives the 988 referral.
4. **Judge.** Each side is put in a band: calm, check-in or support. The voice is banded around its validated threshold. The judge keeps the higher band, so neither side can overrule distress from the other.
Before any of this, Monitor asks the person to **calibrate**: read three to five fixed neutral sentences (from RAVDESS and CREMA-D) in a calm voice. The per-layer, per-channel mean and std of those readings' WavLM features are stored on disk per browser session (`artifacts/enrollments/`), and later clips from that session are z-scored with them before the head. Without this a new microphone behaves like a new corpus, and the cross-corpus study shows what that does (specificity near zero). Needs a `voice_head.pt` trained by this version of `app.voice`, which adds the enrolled head.

5. **Support agent.** On check-in or support, a short conversation with state (turn, risk, step, what was shared): reflect, ask what would help, offer one grounding idea or resource, check in, keep listening. It says it is not a counselor and never shows scores; those stay in the studio views. The agent is a fixed state machine by design: a safety choice that keeps every step predictable and testable, not a claim about conversational AI.

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
| `python -m app.train` | Trains the Monitor model. Actors 01–18 train, 19–20 choose the epoch, 21–24 are the test set (the same split as the Kaggle run). | `artifacts/audio_branch.pt`, `artifacts/training_report.json` |
| `python -m app.cv` | Leave-speakers-out cross-validation: six folds over all 24 RAVDESS actors, or `--corpus crema-d` for five folds over the 91 CREMA-D speakers. Runs the CNN, an eGeMAPS + logistic regression baseline, and a frozen WavLM-base-plus encoder (default head four-attentive, the proposal's design; the live head is `--head binary-mean`). Each fold is scored raw (threshold 0.5) and calibrated (temperature and threshold fitted on validation speakers), with ECE, and models are compared with paired tests over folds. `--only cnn\|egemaps\|wavlm` runs one model; `--head binary-mean` gives the first WavLM head; `--encoder facebook/wav2vec2-base` swaps the encoder. | `artifacts/cv_report.json`, `artifacts/cv_report_crema-d.json`, `artifacts/ssl_cache/` |
| `python -m app.study` | Cross-corpus study, RAVDESS ↔ CREMA-D (needs `data/crema-d/`), for the CNN, eGeMAPS and WavLM on the same speaker splits. `--models wavlm egemaps` picks models; `--encoder` and `--head` work as in `app.cv`; `--norm corpus\|speaker` standardizes features per corpus or speaker with unlabeled audio. Every run is also calibrated, with a speaker-bootstrap 95% interval. | `artifacts/cross_corpus_report.json` |
| `python -m app.voice` | Trains the live WavLM voice model: RAVDESS actors 01–18 plus CREMA-D (when present) train, actors 19–20 and nine CREMA-D speakers choose the epoch, temperature and threshold, actors 21–24 are scored once. A second, calm-enrolled head is trained on features z-scored per speaker by five of their own calm clips (left out of every split); the studio uses it once a person has calibrated. Every epoch and every test clip go into `voice_report.json`; `--progress file.jsonl` also logs each epoch as it finishes. Monitor uses the head as soon as it is in `artifacts/`. | `artifacts/voice_head.pt`, `artifacts/voice_report.json` |
| `python -m app.safety_eval` | Runs the crisis screen and the agent on the 100-message safety set (`eval/safety_set.jsonl`) and checks the targets: every explicit message gets 988, at least 90% of indirect messages are flagged, at most 10% of harmless ones are flagged high. Exits 1 on a miss. `--no-model` tests the keyword floor alone. | `artifacts/safety_report.json` |
| `python -m app.reader_eval` | Compares the phrase list, the language-model reader and a GoEmotions RoBERTa on 200 labeled sentences (`eval/reader_sentences.jsonl`), overall and per group. | `artifacts/reader_report.json` |
| `python -m app.enroll_eval` | Calm enrollment on held-out speakers: none vs full speaker standardization vs calm enrollment (5 clips), on RAVDESS 21–24 and both cross-corpus directions, per speaker mean ± std, plus a 1/3/10-clip sweep. Cached features only. | `artifacts/enrollment_report.json` |
| `python -m app.pipeline_eval` | End to end: real held-out voice readings paired with the 200 sentences in four cells (calm/distressed text × calm/distressed voice) through voice band, words reader, crisis screen and judge; and the 55 harmless safety messages with calm voices (joint false-alarm rate). | `artifacts/pipeline_report.json` |
| `python -m app.safety_split` | Held-out safety numbers: 60/40 stratified split of the 100 messages and 40 new held-out messages, keyword screen as written and with dev-only rules, saved model run per split, Wilson intervals. `--model` runs the risk model live. | `artifacts/safety_heldout_report.json` |
| `python -m app.asr_eval` | Word error rate of faster-whisper on fixed-sentence clips, calm vs distressed. | `artifacts/asr_report.json` |
| `python -m app.seed_eval` | RAVDESS 6-fold CV of the binary-mean head with seeds 7, 17, 27. | `artifacts/seed_report.json` |
| `python -m app.ece_eval` | ECE at 5, 10 and 15 bins (equal width and equal mass) and a reliability diagram. | `artifacts/ece_report.json`, `artifacts/reliability.png` |
| `python -m app.latency_eval` | Time of each stage of one live reading (ASR, voice, reader, crisis screen, judge). | `artifacts/latency_report.json` |
| `python -m pytest tests` | Unit tests for the judge bands and floor, the crisis screen, the support flow and calibration. No model calls. | |
| `python -m app.compare` | Reads the three reports above and writes one comparison table plus a check against the proposal targets. | `artifacts/comparison.md` |

Labels everywhere: distress = sad, angry, fearful; non-distress = neutral, calm; happy, disgust and surprise are excluded.

### This week's experiments on Kaggle

`kaggle/DistressNet_Experiments.ipynb` runs every experiment that needs the audio, a GPU or a language model, from this branch: the live head with the calm-enrolled head (`app.voice`), `app.enroll_eval`, `app.seed_eval`, the pooling/label ablation (`app.cv --head four-mean` and `binary-attentive`), `app.ece_eval`, `app.asr_eval`, a local Ollama for `app.reader_eval --readers llm`, `app.safety_eval`, `app.safety_split --model` and `app.pipeline_eval`, and `app.latency_eval` on GPU and CPU. It packs `distressnet_week_results.zip`, which **Load Kaggle results** installs.

### Training on Kaggle

The saved models and reports in `backend/artifacts/` came from one Kaggle GPU run of `kaggle/DistressNet_Kaggle.ipynb` (about 70 minutes on a T4). It clones `main` of this repo, attaches the Kaggle datasets `uwrfkaggler/ravdess-emotional-speech-audio` and `ejlok1/cremad` (nothing to download), runs `app.cv`, `app.study`, `app.voice`, `app.train`, `app.reader_eval` and `app.compare` in that order, prints the new numbers next to the saved ones, and packs `distressnet_results.zip`. `kaggle/distress-net-wavelm.ipynb` and `kaggle/distressnet-run-1.ipynb` are the earlier runs, kept for reference.

From the studio: **Train on GPU** on the Training page opens that notebook in Kaggle. Turn on a GPU and Internet under Session options, then **Save Version → Save & Run All**. When the version finishes, download `distressnet_results.zip` from its Output tab and choose **Load Kaggle results** on the Training page; the new head goes live on Monitor and the reports update. The button opens https://www.kaggle.com/code/sarthakpokhrel/notebook5cd1df997e; `DISTRESSNET_KAGGLE_URL` points it at another notebook.

The WavLM model downloads its weights from Hugging Face the first time it runs (about 380 MB; on Kaggle, turn Internet on). Every clip is embedded once and cached, so later runs only train the head.

Checkpoints and hyperparameters are chosen on validation speakers by balanced accuracy, never on test speakers. Reports include balanced accuracy, macro F1, AUROC, specificity and the F1 of always predicting distress, which is the baseline to beat.
