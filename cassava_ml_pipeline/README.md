# Cassava Leaf Disease Classification — ML Pipeline

**African Leadership University — Machine Learning Pipeline Summative**

## Video Demo
📺 [YouTube Link — ADD HERE]
*(Demo covers: single-image prediction, bulk upload, and triggering retraining, camera on.)*

## Live URLs
- API (Swagger docs): `http://<deployed-host>:8000/docs` — ADD DEPLOYED URL HERE
- UI (Streamlit): `http://<deployed-host>:8501` — ADD DEPLOYED URL HERE

## Project Description

This project implements an end-to-end ML pipeline for classifying cassava
leaf images into 5 categories:

| Label | Class |
|---|---|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

Cassava is a staple crop across sub-Saharan Africa, and these diseases cause
significant yield loss. The pipeline lets a field worker photograph a leaf,
get an instant diagnosis, and lets an admin bulk-upload newly labeled images
to continuously improve the model over time.

**Pipeline stages** (see `notebook/cassava_classification.ipynb` for the full
offline process):
1. **Data acquisition** — Kaggle's Cassava Leaf Disease Classification dataset
2. **Data processing** — augmentation, rescaling, train/val/test split
3. **Model creation** — EfficientNetB0 transfer learning, fine-tuned, regularized
4. **Model testing** — accuracy, macro precision/recall/F1, confusion matrix
5. **Model retraining** — incremental fine-tune trigger on newly uploaded data
6. **API** — FastAPI serving predict / upload / retrain / uptime / visualizations

## Repository Structure

```
cassava_ml_pipeline/
│
├── README.md
├── notebook/
│   └── cassava_classification.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── api/
│   └── main.py              # FastAPI app
├── ui/
│   └── app.py                # Streamlit dashboard
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── cassava_efficientnet.keras
├── locustfile.py             # Load testing
├── Dockerfile                # API container
├── Dockerfile.ui              # UI container
├── docker-compose.yml
└── requirements.txt
```

## Setup Instructions

### 1. Clone and install
```bash
git clone <repo-url>
cd cassava_ml_pipeline
pip install -r requirements.txt
```

### 2. Get the data and train the model
Open `notebook/cassava_classification.ipynb` (Jupyter/Colab) and run all cells.
This downloads the dataset, splits it, trains `EfficientNetB0`, evaluates it,
and saves `models/cassava_efficientnet.keras`.

### 3. Run the API locally
```bash
uvicorn api.main:app --reload --port 8000
```
Swagger docs at `http://localhost:8000/docs`.

### 4. Run the UI locally
```bash
streamlit run ui/app.py
```

### 5. Run everything with Docker
```bash
docker compose up --build
```
- API: `http://localhost:8000`
- UI: `http://localhost:8501`

### 6. Load testing (Locust)
```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```
Open `http://localhost:8089`, set number of users + spawn rate, run against
the API. To compare container scaling, run:
```bash
docker compose up --build --scale api=1
docker compose up --build --scale api=2
docker compose up --build --scale api=4
```
and repeat the Locust run at each scale, recording results below.

## Results from Flood Request Simulation

| Docker containers (`api` replicas) | Users | Spawn rate | Median latency (ms) | 95th percentile (ms) | RPS | Failures |
|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — |

*(Fill in after running Locust at each scale — export CSVs with `--csv=locust_results/run_N` and summarize here.)*

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/uptime` | Service uptime |
| POST | `/predict` | Predict a single uploaded image |
| POST | `/upload` | Bulk upload labeled images for retraining |
| POST | `/retrain` | Trigger retraining on current training data |
| GET | `/retrain/status` | Poll retraining job status |
| GET | `/metrics` | Latest evaluation metrics |
| GET | `/visualizations/class-distribution` | Class balance chart data |
| GET | `/visualizations/image-stats` | Per-class brightness/resolution stats |

## Model Evaluation Summary

See `notebook/cassava_classification.ipynb` Section 5 for the full
confusion matrix and classification report. Key metrics (fill in after
training):

- Accuracy: —
- Precision (macro): —
- Recall (macro): —
- F1 (macro): —
