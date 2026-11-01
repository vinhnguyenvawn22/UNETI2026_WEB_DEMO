"""Đường dẫn và các tham số thường cần sửa. Đọc README trước khi chạy."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

SEED = 42
TUNING_SEED = 43
CV_SEED = 20261027
N_FOLDS = 3
N_JOBS = min(8, os.cpu_count() or 1)  # Có thể đặt 2 hoặc 4 nếu CPU/RAM hạn chế.
EXPECTED_TRAIN_ROWS = 1200000
EXPECTED_TEST_ROWS = 3000
MIN_SECONDS = 1.0

BASE_TREES = 5707
LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.025,
    "num_leaves": 63,
    "min_child_samples": 562,
    "max_depth": -1,
    "reg_lambda": 1.841796842892308,
    "reg_alpha": 0.0,
    "feature_fraction": 0.7617241098743734,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_bin": 255,
}
RIDGE_ALPHA = 100.0
RF_TREES = 120
RF_MAX_LEAF_NODES = 512

OPTUNA_TOTAL_TRIALS = 20  # Tổng trial COMPLETE mong muốn, không phải số thêm.
OPTUNA_TIMEOUT_SECONDS = (
    7200  # Dừng nhận trial mới sau khoảng này; không cắt trial đang chạy.
)
MAX_TREES = 10000
EARLY_STOPPING = 200
CV_TOP_TRIALS = 2  # Baseline + tối đa 2 trial tốt để đối chiếu CV.

TEAM_CODE = ""
MEMBERS = ["", "", ""]  # Điền 3 họ tên trước khi xuất báo cáo cuối.


def ensure_directories():
    for folder in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def csv_path(kind):
    stem = "UNETI2026_TRAIN_CHUNG_1200000" if kind == "train" else "UNETI01_TEST_3000"
    for name in [stem + "(1).csv", stem + ".csv"]:
        path = DATA_DIR / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Đặt {stem}(1).csv vào {DATA_DIR}")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def data_signature():
    return {
        "train_sha256": file_hash(csv_path("train")),
        "test_sha256": file_hash(csv_path("test")),
        "features_sha256": file_hash(PROJECT_DIR / "features.py"),
    }


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def versions():
    result = {}
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "optuna",
        "streamlit",
    ]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "chưa cài"
    return result
