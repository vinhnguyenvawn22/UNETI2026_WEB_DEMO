# UNETI01 — Mã nguồn để tự chạy từ đầu

Gói này chỉ gồm mã nguồn. models/ và outputs/ ban đầu để trống. Các model, điểm số, biểu đồ, SQLite và CSV sẽ do bạn tạo khi chạy notebook. Không có model huấn luyện sẵn hoặc kết quả cũ được tự động nạp.

## Cấu trúc

| Vị trí | Chức năng |
|---|---|
| data/ | Đặt trực tiếp hai CSV gốc |
| notebooks/01_EDA.ipynb | Đọc dữ liệu, kiểm tra, ba biểu đồ và nhận xét |
| notebooks/02_Preprocessing.ipynb | Tạo 49 đặc trưng, lưu các NPY |
| notebooks/03_Train_3_Models.ipynb | Chia tập và chạy Ridge, Random Forest, LightGBM |
| notebooks/04_Optimize_LightGBM.ipynb | Optuna, CV, phân tích lỗi và diagnostic lịch sử |
| notebooks/05_Final_Submission.ipynb | Fit LightGBM toàn Train, CSV, kiểm tra nạp lại, PDF 3 trang |
| config.py | Đường dẫn, seed và các tham số cần sửa |
| features.py | Công thức đặc trưng, metric và suy luận dùng chung |
| app.py | Ứng dụng Streamlit |
| models/ | Model do bạn huấn luyện |
| outputs/ | Biểu đồ, split, điểm, SQLite, submission và báo cáo |

## 1. Chuẩn bị trên Drive

Giải nén để có MyDrive/UNETI01. Đặt:

- data/UNETI2026_TRAIN_CHUNG_1200000(1).csv
- data/UNETI01_TEST_3000(1).csv

Tên không có (1) cũng được chấp nhận. Không nén dữ liệu. Giữ nguyên thứ tự dòng CSV.

Mỗi notebook có ô chọn PROJECT_DIR. Mặc định là /content/drive/MyDrive/UNETI01. Chỉ sửa dòng này nếu đặt ở nơi khác. Các đường dẫn còn lại suy ra từ vị trí config.py.

## 2. Cài thư viện

Mở một notebook bằng Colab, gắn Drive, rồi chạy một lần:

    %pip install -r /content/drive/MyDrive/UNETI01/requirements.txt

Nếu runtime đã import NumPy/Pandas phiên bản khác, khởi động lại runtime sau khi cài, rồi chạy lại từ ô đầu. Môi trường kiểm tra mã nguồn là Python 3.12.

## 3. Chạy lần lượt 01 → 05

Chạy các ô từ trên xuống trong từng notebook. Có thể mở mỗi notebook ở một phiên riêng: các bước đọc file do bước trước lưu, không phụ thuộc biến còn trong RAM.

**01:** kiểm tra dữ liệu và tạo ba biểu đồ. Không tự xóa dòng khó hoặc thay nhãn.

**02:** tạo đặc trưng bằng công thức xác định, không fit nhãn. Cache gắn hash của CSV và features.py. NaN được giữ để imputer của Ridge/RF fit riêng trên training; LightGBM xử lý ở cây.

**03:** tạo split seed 42 giữ 20%, seed 43 lấy tuning 15% từ phần phát triển. Chạy ba mô hình thực sự trên cùng tuning. Ridge và Random Forest được dùng làm đối chiếu; LightGBM là mô hình chính để tối ưu. Nhãn học log1p; mọi dự đoán chuyển về giây bằng expm1, chặn tối thiểu 1 giây nhất quán.

**04:** Optuna có SQLite, lưu sampler và kết quả từng trial. CV đối chiếu baseline với các trial triển vọng trên cùng 3 fold seed 20261027; lưu sau mỗi fold. Chọn cấu hình trước khi chấm lại validation lịch sử 20%. Các số tuning/CV đều dùng lựa chọn, không phải điểm Test chính thức.

**05:** huấn luyện một LightGBM trên toàn bộ Train; xuất đúng 3.000 dòng Test; kiểm tra nạp lại model và suy luận một dòng; tạo PDF 3 trang từ số liệu đã chạy.

Không dùng trip_id làm đặc trưng, thông tin sau chuyến đi hoặc nhãn Test ngoài đề. Không ensemble/stacking hay lấy trung bình các model fold.

## 4. Các tham số thường sửa trong config.py

| Tham số | Ý nghĩa |
|---|---|
| N_JOBS | Luồng CPU, giảm xuống 2–4 nếu cần |
| BASE_TREES | Số cây LightGBM tham khảo, mặc định 5707 |
| RF_TREES | Số cây Random Forest, mặc định 120 |
| OPTUNA_TOTAL_TRIALS | Tổng trial COMPLETE cần có, mặc định 20 |
| OPTUNA_TIMEOUT_SECONDS | Ngân sách thời gian mỗi phiên tìm kiếm |
| MAX_TREES, EARLY_STOPPING | Giới hạn cây và patience khi tuning |
| CV_TOP_TRIALS | Số trial tốt đưa vào CV, ngoài baseline |
| TEAM_CODE, MEMBERS | Thông tin dùng trong báo cáo |

Để chạy thêm buổi sau trước khi khóa lựa chọn: mở lại notebook 04. Chỉ các trial còn thiếu mới chạy; có thể tăng OPTUNA_TOTAL_TRIALS. Timeout không cắt giữa trial đang chạy.

Sau khi selection.json đã được ghi, notebook 04 dùng lại lựa chọn và diagnostic, không tự tối ưu tiếp dựa vào validation đã xem. Muốn nghiên cứu tiếp sau thời điểm này, tạo bản sao thư mục dự án với models/ và outputs/ trống; validation lịch sử vẫn là tập đã được xem, không trở thành holdout mới.

Nếu thay dữ liệu, features.py hoặc các cấu hình cốt lõi sau khi chạy, dùng thư mục chạy mới. Không lấy cache/model của hai định nghĩa khác nhau trộn với nhau. Khi chỉ điền họ tên, mở lại kernel và chạy lại notebook 05; model hoàn tất có thể được dùng lại, không cần fit lại vì thông tin thành viên không ảnh hưởng cấu hình mô hình.

## 5. Điểm tham khảo và giới hạn

Cấu hình mặc định và công thức 49 đặc trưng xuất phát từ phương án đã có kết quả tuning khoảng 0,3347 trong lịch sử dự án. Đây không phải điểm được bảo đảm khi chạy lại hoặc trên Test. File nguồn không điền sẵn điểm này vào output.

Code tạo split mới theo seed đã thống nhất; không nhập chỉ mục/checkpoint cũ. Chỉ so sánh trực tiếp với kết quả trước khi xác minh cùng dữ liệu, thứ tự dòng và chỉ mục.

Validation 20% đã được xem trong lịch sử dự án. Điểm chấm lại chỉ là diagnostic sau khi khóa. Không có holdout độc lập mới để xác nhận toàn bộ quá trình nghiên cứu.

Chọn A100 không tự làm LightGBM wheel chạy GPU. Mặc định dùng CPU; mỗi lần chỉ chạy một tác vụ huấn luyện nặng. Optuna/CV trên 1,2 triệu dòng có thể mất nhiều giờ và nhiều phiên. Phần cứng nhanh hơn không bảo đảm RMSLE dưới 0,32 hoặc 0,30.

## 6. Mở ứng dụng

Sau khi notebook 05 hoàn tất, chạy từ thư mục UNETI01 trên máy có thể mở Streamlit:

    python -m pip install -r requirements.txt
    streamlit run app.py

App cần config.py, features.py và model/metadata do notebook 05 tạo. App không cần đọc CSV Train để dự đoán một chuyến mới. Colab cần môi trường truy cập giao diện thích hợp; gói này không tự mở tunnel hay đưa app lên Internet.

Nếu chưa chạy 05, app sẽ báo chưa có model. App gọi model thật và cùng features.py, không đọc kết quả có sẵn từ submission.

## 7. Đầu ra sau khi chạy

- models/lightgbm_final.txt và models/final_model_info.json: model cuối và metadata.
- models/Ridge.joblib, models/RandomForest.joblib, models/LightGBM.txt: model đối chiếu.
- outputs/three_models_scores.csv: so sánh tuning ba mô hình.
- outputs/cv_folds.csv, cv_comparison.csv: điểm từng fold, OOF và so sánh ghép cặp.
- outputs/selection.json: cấu hình đã khóa.
- outputs/historical_validation.json: diagnostic lịch sử.
- outputs/UNETI01_submission_lightgbm.csv: file Test đúng hai cột.
- outputs/UNETI01_BaoCao.pdf: báo cáo ba trang.

Điền ba họ tên trong config.py trước khi tạo báo cáo nộp chính thức. Theo thông báo BTC, hoàn thiện tên file và cách nộp.

## 8. Video demo 2–3 phút

Quay màn hình thật: khởi chạy app, nhập một chuyến với khoảng cách 2,5 km và giờ 8, nhấn dự đoán; đổi khoảng cách thành 10 km, giờ 18 và điểm đến phù hợp, nhấn lại và quay rõ kết quả giây/phút. Không bắt buộc quay mặt hoặc triển khai Internet. Chưa có video ghi sẵn trong gói mã nguồn.

## Kiểm tra trước khi bàn giao

Cú pháp và schema notebook được kiểm tra. Các notebook cùng app được chạy kiểm tra trên một tập con dữ liệu thật trong thư mục thử nghiệm riêng, với số cây/trial nhỏ để kiểm tra luồng xử lý và resume. Đây là kiểm thử kỹ thuật, không phải kết quả huấn luyện đầy đủ hoặc điểm RMSLE mới.

File phân phối giữ nguyên cấu hình đầy đủ, toàn bộ output ô code rỗng, models/ và outputs/ trống để bạn tự chạy.
# UNETI2026_WEB_DEMO
