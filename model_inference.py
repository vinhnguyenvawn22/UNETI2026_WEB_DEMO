"""Suy luận với bộ model cuối và model_info.json do train.ipynb xuất."""

from importlib.metadata import version

import joblib

import config as C
from features import RAW_FEATURES, create_features, to_seconds


AVAILABLE_MODELS = ("LightGBM", "Ridge", "RandomForest")


def _read_model_info():
    path = C.MODEL_DIR / "model_info.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Thiếu models/model_info.json. Chép metadata và ba model cuối "
            "từ results/ của train.ipynb vào models/."
        )
    info = C.read_json(path)
    if info["feature_version"] != "numeric49_v1":
        raise ValueError("Phiên bản đặc trưng chưa được app hỗ trợ.")
    if info["raw_features"] != RAW_FEATURES:
        raise ValueError("Các cột dữ liệu gốc không khớp bộ model.")
    if (info["target"] != "trip_duration" or info["target_unit"] != "seconds"
            or info["target_transform"] != "log1p"):
        raise ValueError("Phép biến đổi nhãn không khớp cách dự đoán của app.")
    if info["minimum_seconds"] != C.MIN_SECONDS:
        raise ValueError("MIN_SECONDS không khớp metadata huấn luyện.")
    if info["feature_count"] != len(info["feature_columns"]):
        raise ValueError("Số cột đặc trưng trong metadata không hợp lệ.")
    return info


def _model_path(info, model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")
    filename = info["final_models"][model_name]["file"]
    path = (C.MODEL_DIR / filename).resolve()
    if path.parent != C.MODEL_DIR.resolve():
        raise ValueError("Đường dẫn model phải nằm trong models/.")
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu models/{filename}. Chép từ results/ của train.ipynb.")
    return path


def model_revision(model_name):
    """Đổi khóa cache Streamlit khi thay metadata, model hoặc mã đặc trưng."""
    info = _read_model_info()
    paths = [C.MODEL_DIR / "model_info.json", _model_path(info, model_name),
             C.PROJECT_DIR / "features.py"]
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


def load_selected_model(model_name):
    info = _read_model_info()
    path = _model_path(info, model_name)
    if info["model_sha256"][path.name] != C.file_hash(path):
        raise ValueError(f"{path.name} không khớp model_info.json. Cần chép cùng bộ model.")

    if model_name == "LightGBM":
        import lightgbm as lgb

        # Model dạng text được nạp để suy luận trên CPU, không cần backend CUDA.
        model = lgb.Booster(model_file=str(path))
        if model.feature_name() != info["feature_columns"]:
            raise ValueError("LightGBM không khớp tên hoặc thứ tự đặc trưng.")
        return model, info

    trained_version = info["versions"]["scikit-learn"]
    if version("scikit-learn") != trained_version:
        raise ValueError(
            f"Cần scikit-learn=={trained_version} để nạp {model_name}; "
            f"hiện đang dùng {version('scikit-learn')}. Cập nhật requirements.txt."
        )
    model = joblib.load(path)
    if model.n_features_in_ != info["feature_count"]:
        raise ValueError(f"{model_name} không khớp số đặc trưng đã huấn luyện.")
    names = getattr(model, "feature_names_in_", None)
    if names is not None and list(names) != info["feature_columns"]:
        raise ValueError(f"{model_name} không khớp thứ tự đặc trưng.")
    # Pipeline Random Forest có thể lưu số luồng của máy train lớn hơn máy chạy app.
    if "randomforestregressor" in model.named_steps:
        model.named_steps["randomforestregressor"].n_jobs = C.N_JOBS
    return model, info


def predict_selected_model(raw, model_name, loaded=None):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")
    model, info = load_selected_model(model_name) if loaded is None else loaded
    x = create_features(raw)
    if list(x.columns) != info["feature_columns"]:
        raise ValueError("Các cột đầu vào không khớp đặc trưng đã huấn luyện.")
    if model_name == "LightGBM":
        prediction = model.predict(x, num_threads=C.N_JOBS)
    else:
        # train.ipynb fit các pipeline bằng mảng NumPy, nhãn log1p.
        values = x if hasattr(model, "feature_names_in_") else x.to_numpy()
        prediction = model.predict(values)
    return to_seconds(prediction)
