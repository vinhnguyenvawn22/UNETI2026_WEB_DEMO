"""49 đặc trưng số và hàm đánh giá/suy luận dùng chung. Không dùng trip_id hoặc nhãn."""

import numpy as np
import pandas as pd
import config as C

RAW_FEATURES = [
    "vendor_id",
    "pickup_month",
    "pickup_dayofweek",
    "pickup_hour",
    "pickup_minute_bucket",
    "passenger_count",
    "pickup_lon_approx",
    "pickup_lat_approx",
    "dropoff_lon_approx",
    "dropoff_lat_approx",
    "straight_line_km_approx",
    "store_and_fwd_flag",
]


def create_features(raw):
    x = raw[RAW_FEATURES].copy().reset_index(drop=True)
    x["store_and_fwd_flag"] = x["store_and_fwd_flag"].map({"N": 0, "Y": 1}).fillna(0)
    for c in x:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    distance = x.straight_line_km_approx.clip(lower=0)
    x["log_distance"] = np.log1p(distance)
    lat_scale = np.cos(np.deg2rad((x.pickup_lat_approx + x.dropoff_lat_approx) / 2))
    dx = (x.dropoff_lon_approx - x.pickup_lon_approx) * 111.32 * lat_scale
    dy = (x.dropoff_lat_approx - x.pickup_lat_approx) * 111.32
    x["delta_x_km"] = dx
    x["delta_y_km"] = dy
    x["abs_delta_x_km"] = np.abs(dx)
    x["abs_delta_y_km"] = np.abs(dy)
    x["coordinate_distance"] = np.hypot(dx, dy)
    x["manhattan_distance"] = np.abs(dx) + np.abs(dy)
    # Street-grid rotation is a deterministic transformation, not a recovered location.
    angle = np.deg2rad(29)
    u = dx * np.cos(angle) - dy * np.sin(angle)
    v = dx * np.sin(angle) + dy * np.cos(angle)
    x["grid_manhattan_distance"] = np.abs(u) + np.abs(v)
    x["direction_sin"] = dy / np.maximum(np.hypot(dx, dy), 0.01)
    x["direction_cos"] = dx / np.maximum(np.hypot(dx, dy), 0.01)
    hour = x.pickup_hour + x.pickup_minute_bucket / 60
    dow = x.pickup_dayofweek
    x["hour_fraction"] = hour
    x["week_hour"] = dow * 24 + hour
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    x["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    x["is_weekend"] = (dow >= 5).astype(int)
    x["is_night"] = ((hour < 6) | (hour >= 22)).astype(int)
    x["is_rush"] = (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 20))).astype(
        int
    )
    x["mid_lon"] = (x.pickup_lon_approx + x.dropoff_lon_approx) / 2
    x["mid_lat"] = (x.pickup_lat_approx + x.dropoff_lat_approx) / 2
    x["distance_coordinate_ratio"] = distance / (x.coordinate_distance + 0.5)
    x["distance_coordinate_difference"] = distance - x.coordinate_distance
    x["same_rounded_location"] = (x.coordinate_distance < 0.01).astype(int)
    for name in ["is_weekend", "is_night", "is_rush"]:
        x["distance_" + name] = distance * x[name]
    # Squared distances to airport reference locations: input-only geography.
    for name, lon, lat in [
        ("jfk", -73.78, 40.64),
        ("lga", -73.87, 40.77),
        ("ewr", -74.18, 40.69),
    ]:
        for point in ["pickup", "dropoff"]:
            x[f"{point}_{name}_km"] = np.hypot(
                (x[f"{point}_lon_approx"] - lon) * 84.3,
                (x[f"{point}_lat_approx"] - lat) * 111.32,
            )
    for point in ["pickup", "dropoff"]:
        x[f"{point}_grid_u"] = x[f"{point}_lon_approx"] * 84.3 * np.cos(angle) - x[
            f"{point}_lat_approx"
        ] * 111.32 * np.sin(angle)
        x[f"{point}_grid_v"] = x[f"{point}_lon_approx"] * 84.3 * np.sin(angle) + x[
            f"{point}_lat_approx"
        ] * 111.32 * np.cos(angle)
    return x.replace([np.inf, -np.inf], np.nan).astype("float32")


def to_seconds(pred_log):
    """Quy tắc giống nhau cho mọi mô hình: tối thiểu 1 giây."""
    pred_log = np.asarray(pred_log, dtype=float)
    if not np.isfinite(pred_log).all():
        raise ValueError("Dự đoán log không hữu hạn.")
    prediction = np.expm1(np.maximum(pred_log, np.log1p(C.MIN_SECONDS)))
    if not np.isfinite(prediction).all():
        raise ValueError("Dự đoán giây vượt giới hạn số học.")
    return prediction


def rmsle(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Nhãn hoặc dự đoán không hữu hạn.")
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("RMSLE yêu cầu giá trị không âm.")
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def load_processed():
    """Notebook sau đọc file do notebook 02 tạo, không phụ thuộc RAM phiên trước."""
    info_path = C.OUTPUT_DIR / "features_info.json"
    if not info_path.exists():
        raise FileNotFoundError("Chạy notebook 02_Preprocessing.ipynb trước.")
    info = C.read_json(info_path)
    if info["signature"] != C.data_signature():
        raise ValueError(
            "Dữ liệu hoặc features.py đã thay đổi. Tạo lần chạy mới từ bước 01."
        )
    x = np.load(C.OUTPUT_DIR / "X_train.npy", mmap_mode="r", allow_pickle=False)
    test_x = np.load(C.OUTPUT_DIR / "X_test.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(C.OUTPUT_DIR / "y_seconds.npy", allow_pickle=False)
    train_ids = np.load(C.OUTPUT_DIR / "train_ids.npy", allow_pickle=False)
    test_ids = np.load(C.OUTPUT_DIR / "test_ids.npy", allow_pickle=False)
    assert x.shape == tuple(info["train_shape"]) and test_x.shape == tuple(
        info["test_shape"]
    )
    assert len(y) == len(train_ids) == len(x)
    assert len(test_ids) == len(test_x)
    return x, test_x, y, train_ids, test_ids, info


def load_final_model():
    import lightgbm as lgb

    model_path = C.MODEL_DIR / "lightgbm_final.txt"
    metadata_path = C.MODEL_DIR / "final_model_info.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Chạy notebook 05_Final_Submission.ipynb để huấn luyện model cuối."
        )
    info = C.read_json(metadata_path)
    if info["features_sha256"] != C.file_hash(C.PROJECT_DIR / "features.py"):
        raise ValueError("features.py khác phiên bản đã huấn luyện.")
    if info["model_sha256"] != C.file_hash(model_path):
        raise ValueError("Tệp model không khớp metadata.")
    if info["minimum_seconds"] != C.MIN_SECONDS:
        raise ValueError("MIN_SECONDS đã thay đổi sau khi huấn luyện.")
    model = lgb.Booster(model_file=str(model_path))
    assert model.feature_name() == info["columns"]
    return model, info


def predict_duration(raw, loaded=None):
    model, info = load_final_model() if loaded is None else loaded
    x = create_features(raw)
    assert list(x) == info["columns"]
    return to_seconds(model.predict(x, num_threads=C.N_JOBS))
