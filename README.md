## Kidney Disease Classification (MLOps: DVC + MLflow)

End-to-end deep learning pipeline for kidney CT scan classification with:
- **DVC** for reproducible pipelines (`dvc.yaml`)
- **MLflow** for experiment tracking (defaults to local `./mlruns`)
- **Flask** app for inference (`app.py`)

## How to run (local)

### Clone

```bash
git clone https://github.com/Karthik016996148/Kidney-Disease-Classification-MLOps.git
cd Kidney-Disease-Classification-MLOps
```

### Setup environment

```bash
conda create -n cnncls python=3.10 -y
conda activate cnncls
pip install -r requirements.txt
```

### Train (pipeline)

- **Option A (recommended)**: run the full DVC pipeline

```bash
dvc repro
```

- **Option B**: run the python pipeline script directly

```bash
python main.py
```

### Track experiments with MLflow

By default this project logs to a local store: **`file:./mlruns`**.

```bash
mlflow ui --backend-store-uri ./mlruns
```

To use a remote tracking server, set environment variables (copy `env.example` and fill values):
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME` (optional; depends on your server)
- `MLFLOW_TRACKING_PASSWORD` (optional; depends on your server)

To disable MLflow logging entirely:

```bash
set ENABLE_MLFLOW_LOGGING=false
```

## Run the web app

```bash
python app.py
```

Then open `http://localhost:8080`.

## Notes

- **No secrets are committed**: credentials must be supplied via environment variables.
- **MLflow docs**: `https://mlflow.org/docs/latest/index.html`

