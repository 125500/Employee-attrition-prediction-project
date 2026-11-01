from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = ROOT / "models" / "attrition_pipeline.joblib"
REPORTS_DIR = ROOT / "reports"

TARGET = "Attrition"
# constant or ID-only columns, no predictive value
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
