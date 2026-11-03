"""Chạy từ thư mục dự án: python -m unittest discover -s tests -v."""

import ast
import json
import unittest

import numpy as np
import pandas as pd

import config as C
from features import RAW_FEATURES, create_features, to_seconds
from model_inference import AVAILABLE_MODELS, load_selected_model, predict_selected_model


def sample_trips():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "vendor_id": rng.integers(1, 3, n),
        "pickup_month": rng.integers(1, 13, n),
        "pickup_dayofweek": rng.integers(0, 7, n),
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_minute_bucket": rng.choice([0, 15, 30, 45], n),
        "passenger_count": rng.integers(0, 7, n),
        "pickup_lon_approx": rng.uniform(-74.2, -73.7, n).round(2),
        "pickup_lat_approx": rng.uniform(40.55, 40.9, n).round(2),
        "dropoff_lon_approx": rng.uniform(-74.2, -73.7, n).round(2),
        "dropoff_lat_approx": rng.uniform(40.55, 40.9, n).round(2),
        "straight_line_km_approx": rng.choice([0, 0.5, 2.5, 10, 35], n),
        "store_and_fwd_flag": rng.choice(["N", "Y"], n),
    })


class ModelCompatibilityTests(unittest.TestCase):
    def test_notebook_feature_and_inverse_transform_parity(self):
        notebook_path = C.PROJECT_DIR / "train.ipynb"
        if not notebook_path.exists():
            self.skipTest("train.ipynb không có trong bản triển khai")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        namespace = {"np": np, "pd": pd, "RAW_FEATURES": RAW_FEATURES,
                     "MIN_SECONDS": C.MIN_SECONDS}
        found = set()
        # Chỉ chạy định nghĩa hai hàm; không chạy đọc dữ liệu/train/CUDA.
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef) and node.name in {"create_features", "to_seconds"}:
                    exec(compile(ast.Module(body=[node], type_ignores=[]),
                                 "<train.ipynb functions>", "exec"), namespace)
                    found.add(node.name)
        self.assertEqual(found, {"create_features", "to_seconds"})
        raw = sample_trips()
        raw.loc[0, ["dropoff_lon_approx", "dropoff_lat_approx"]] = raw.loc[
            0, ["pickup_lon_approx", "pickup_lat_approx"]].to_numpy()
        raw.loc[1, "pickup_lon_approx"] = np.nan
        raw.loc[2, "straight_line_km_approx"] = -1
        raw.loc[3, "store_and_fwd_flag"] = "unknown"
        pd.testing.assert_frame_equal(create_features(raw), namespace["create_features"](raw),
                                      check_exact=True)
        logs = np.array([-2, 0, np.log1p(C.MIN_SECONDS), 6, 10])
        np.testing.assert_array_equal(to_seconds(logs), namespace["to_seconds"](logs))

    def test_final_artifacts_predict_like_notebook(self):
        raw = sample_trips()
        x = create_features(raw).to_numpy()
        for name in AVAILABLE_MODELS:
            with self.subTest(model=name):
                loaded = load_selected_model(name)
                model, info = loaded
                self.assertEqual(info["feature_columns"], list(create_features(raw)))
                # Notebook dùng NumPy cho cả ba model và áp dụng expm1 một lần.
                kwargs = {"num_threads": C.N_JOBS} if name == "LightGBM" else {}
                expected = to_seconds(model.predict(x, **kwargs))
                actual = predict_selected_model(raw, name, loaded)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue((actual >= C.MIN_SECONDS).all())


if __name__ == "__main__":
    unittest.main()
