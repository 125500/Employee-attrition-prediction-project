# Employee Attrition Prediction

Predict which employees are likely to leave a company, explain why, and serve the
predictions through a FastAPI endpoint so HR can act early.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

## Data
Download the IBM HR Analytics Employee Attrition dataset from Kaggle and save it as:
`data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

## Run
## How to run (Python 3.13 tested)

```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m src.train          # trains the model (about a minute)
python -m src.explain        # optional: SHAP plots in reports/
uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000 in the browser.
Run all commands from the project root (the folder containing `api/` and `src/`).

## Project structure
```
data/        raw dataset (not committed)
notebooks/   EDA and experiments
src/         config, training code
api/         FastAPI app
models/      saved pipeline
reports/     plots and results
```

## Results
_(fill in after training: model comparison table, SHAP plots, key findings)_
