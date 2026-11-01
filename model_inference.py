"""Nạp model cho app, tách khỏi features.py đã dùng khi huấn luyện."""

import joblib

import config as C
from features import create_features, load_final_model, to_seconds


AVAILABLE_MODELS = ("LightGBM", "Ridge", "RandomForest")


def load_selected_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")
    if model_name == "LightGBM":
        return load_final_model()

    info = C.read_json(C.MODEL_DIR / "final_model_info.json")
    if info["features_sha256"] != C.file_hash(C.PROJECT_DIR / "features.py"):
        raise ValueError("features.py khác phiên bản đã huấn luyện.")
    if info["minimum_seconds"] != C.MIN_SECONDS:
        raise ValueError("MIN_SECONDS đã thay đổi sau khi huấn luyện.")
    path = C.MODEL_DIR / f"{model_name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path.name}. Chạy notebook 03 trước.")
    model = joblib.load(path)
    if model.n_features_in_ != len(info["columns"]):
        raise ValueError(f"{model_name} không khớp số đặc trưng đã huấn luyện.")
    names = getattr(model, "feature_names_in_", None)
    if names is not None and list(names) != info["columns"]:
        raise ValueError(f"{model_name} không khớp thứ tự đặc trưng.")
    return model, info


def predict_selected_model(raw, model_name, loaded=None):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")
    model, info = load_selected_model(model_name) if loaded is None else loaded
    x = create_features(raw)
    if list(x.columns) != info["columns"]:
        raise ValueError("Các cột đầu vào không khớp đặc trưng đã huấn luyện.")
    if model_name == "LightGBM":
        prediction = model.predict(x, num_threads=C.N_JOBS)
    else:
        # Notebook 03 fit các pipeline bằng mảng NumPy, nhãn log1p.
        values = x if hasattr(model, "feature_names_in_") else x.to_numpy()
        prediction = model.predict(values)
    return to_seconds(prediction)
