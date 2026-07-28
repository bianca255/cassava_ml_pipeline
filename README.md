# Cassava Leaf Disease Classification 

## Video Demo
📺 [YouTube Link — ADD HERE]
*(Demo covers: single-image prediction, bulk upload, and triggering retraining, camera on.)*

## Live URLs
- API (Swagger docs): https://cassava-api-0liy.onrender.com/docs
- UI (Streamlit): https://cassava-ui.onrender.com

*Note: both services run on Render's free tier and will spin down after ~15 minutes of
inactivity. The first request after idling can take 30–60 seconds while the instance wakes up.*

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
1. **Data acquisition** — Kaggle Cassava Leaf Disease Classification dataset (Datasets mirror)
2. **Data processing** — augmentation, EfficientNet-correct preprocessing, train/val/test split
3. **Model creation** — EfficientNetB0 transfer learning, partially fine-tuned, regularized
4. **Model testing** — accuracy, macro precision/recall/F1, confusion matrix
5. **Model retraining** — incremental fine-tune trigger on newly uploaded data
6. **API** — FastAPI serving predict / upload / retrain / uptime / visualizations

**Engineering note:** an early version of the pipeline used `rescale=1./255` to normalize
images before feeding them to EfficientNetB0. This was a bug — EfficientNetB0 in
`tf.keras.applications` already includes its own internal normalization and expects raw
pixel values, so pre-scaling on top of that corrupted the pretrained ImageNet features and
caused the model to collapse to predicting a single class regardless of input. Switching to
EfficientNet's own `preprocess_input` function (see `src/preprocessing.py`) fixed this and
took the model from ~20–33% accuracy (collapsed/near-random) to ~72% accuracy with genuine
discrimination across all 5 classes.

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
│   └── cassava_efficientnet.h5
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
and saves `models/cassava_efficientnet.h5`.

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

Load testing was run with Locust against the live deployed API
(`https://cassava-api-0liy.onrender.com`) rather than local Docker replicas, since Render's
free tier does not support horizontal scaling (that requires a paid plan). The table below
reflects single-instance performance under concurrent load.

| Endpoint | Requests | Failures | Median (ms) | 95th %ile (ms) | RPS |
|---|---|---|---|---|---|
| POST /predict | 395 | 5 | 4000 | 8900 | 0.9 |
| GET /uptime | 153 | 1 | 6300 | 12000 | 0.4 |
| GET /visualizations/class-distribution | 85 | 0 | 6200 | 11000 | 0.2 |
| **Aggregated** | **700** | **72*** | **4700** | **11000** | **1.6** |

\* Most aggregated failures came from `/metrics` returning 404 during an earlier test run,
before `models/latest_metrics.json` was committed to the repo — this has since been fixed
and confirmed working. `/predict`, the core endpoint, had a 98.7% success rate.

**Test conditions:** 10 concurrent users, ramp-up rate of 2 users/second, run for ~2 minutes
against Render's free-tier instance (shared CPU, 512MB RAM, no GPU). Latencies are elevated
compared to what a paid/GPU-backed instance would show — this is expected given CPU-only
TensorFlow inference on a resource-constrained free tier, and cold-start effects if the
instance had recently spun down from inactivity.

To reproduce locally:
```bash
pip install locust
locust -f locustfile.py --host=https://cassava-api-0liy.onrender.com
```
Open `http://localhost:8089`, set number of users + spawn rate, and start the swarm.

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

See `notebook/cassava_classification.ipynb` Section 5 for the full confusion matrix and
classification report. Evaluated on the full held-out test set (4,277 images):

- **Accuracy: 71.57%**
- **Precision (macro): 60.71%**
- **Recall (macro): 65.15%**
- **F1 (macro): 60.46%**
- Loss: 0.9133

Per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cassava Bacterial Blight (CBB) | 42.9% | 71.0% | 53.5% | 217 |
| Cassava Brown Streak Disease (CBSD) | 64.0% | 48.7% | 55.3% | 437 |
| Cassava Green Mottle (CGM) | 63.8% | 52.2% | 57.4% | 477 |
| Cassava Mosaic Disease (CMD) | 95.3% | 78.1% | 85.8% | 2,631 |
| Healthy | 37.6% | 75.7% | 50.2% | 515 |

CMD — the majority class by a wide margin — is predicted very well. The minority classes
(CBB, Healthy) show lower precision (more false positives) but reasonable recall (the model
catches most true cases), a pattern consistent with the class imbalance visualized in the
notebook's class-distribution chart. Class-weighted loss was used during training to
partially offset this imbalance.