# Chạy app với model từ train.ipynb

Notebook mới lưu kết quả vào `results/`. Sau khi chạy xong các ô huấn luyện,
kiểm tra nạp lại và xuất metadata, chép cùng lúc bốn tệp sau vào `models/`:

- `model_info.json`
- `lightgbm_final.txt`
- `Ridge_final.joblib`
- `RandomForest_final.joblib`

App đọc `model_info.json`, kiểm tra SHA-256 từng model, tên/thứ tự 49 đặc trưng
`numeric49_v1`, phép biến đổi nhãn và phiên bản scikit-learn cho hai pipeline.
Không dùng `final_model_info.json`, `Ridge.joblib` hoặc `RandomForest.joblib` của bản cũ.

LightGBM dùng model cuối được ghi trong metadata; suy luận từ tệp text chạy được
trên CPU dù quá trình huấn luyện dùng CUDA. Ridge và Random Forest đã chứa imputer
và scaler (nếu có); app không fit thêm bước xử lý nào. Cả ba dự đoán nhãn log1p,
đổi về giây bằng expm1 và chặn tối thiểu theo metadata.

## Môi trường và triển khai

`requirements.txt` đã cập nhật NumPy, pandas, scikit-learn, LightGBM và joblib
theo `models/model_info.json` hiện tại. Khi train ở môi trường khác, đối chiếu lại
trường `versions` trước khi triển khai, đặc biệt giữ đúng phiên bản scikit-learn.

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Trên Streamlit Cloud, dùng Python 3.11 và cập nhật `app.py`, `model_inference.py`,
`requirements.txt`, `features.py`, `config.py` cùng bốn tệp trong `models/` lên GitHub.
Chờ cài đặt hoàn tất rồi reboot app. App đổi khóa cache khi model/metadata thay đổi.
Không cần đưa dữ liệu Train hoặc notebook lên Cloud để chạy dự đoán.

## Kiểm tra trước khi thay model

    python -m unittest discover -s tests -v

Bài kiểm tra đối chiếu hai hàm tạo đặc trưng/chuyển đơn vị giữa `train.ipynb`
và app, rồi nạp ba model thật để so sánh đường suy luận. Phép đối chiếu notebook
được bỏ qua nếu không có notebook trong bản triển khai. Không chạy huấn luyện lại.

Nếu thay công thức đặc trưng trong notebook, phải cập nhật `features.py`, tăng
`feature_version` trong notebook và bổ sung hỗ trợ tương ứng ở `model_inference.py`.
Chỉ giữ nguyên tên hoặc số lượng cột không đủ đảm bảo công thức tương thích.
