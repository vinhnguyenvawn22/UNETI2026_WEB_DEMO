
"""Chạy bằng: streamlit run app.py"""

import numpy as np
import pandas as pd
import streamlit as st

from model_inference import (
    AVAILABLE_MODELS,
    load_selected_model,
    predict_selected_model,
)

st.set_page_config(
    page_title="Dự đoán thời gian taxi",
    page_icon="🚕",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {background: var(--background-color);}
    .block-container {max-width: 1180px; padding-top: 2.5rem;}
    .hero {background: linear-gradient(115deg, #102c3c, #18636a);
        color: white; padding: 2.3rem; border-radius: 24px; margin-bottom: 1.6rem;}
    .hero .eyebrow {color: #9ee4d0; font-size: .78rem; font-weight: 700;
        letter-spacing: .18em; margin-bottom: .8rem;}
    .hero h1 {color: white; font-size: clamp(1.8rem, 4vw, 2.7rem);
        padding: 0; margin-bottom: .8rem;}
    .hero p {color: #d7e9ed; max-width: 650px; margin: 0; line-height: 1.7;}
    .hero .badge {display: inline-block; margin-top: 1.3rem; padding: .4rem .8rem;
        border: 1px solid #78b6ad; border-radius: 30px; font-size: .8rem;}
    [data-testid="stForm"] {border-radius: 18px; padding: 1.5rem;}
    [data-testid="stMetric"] {background: var(--secondary-background-color);
        border-radius: 16px; padding: 1.2rem; border: 1px solid #78999b33;}
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 12px; min-height: 3rem; font-weight: 600;}
    @media (max-width: 640px) {
        .block-container {padding-top: 1.2rem;}
        .hero {padding: 1.5rem; border-radius: 18px;}
    }
    </style>
    <div class="hero">
      <div class="eyebrow">UNETI01 / TAXI TIME</div>
      <h1>Chủ động thời gian cho mỗi chuyến đi.</h1>
      <p>Nhập thông tin chuyến taxi để ước tính thời gian di chuyển,
      hoặc khám phá dự đoán từ ba mô hình học máy.</p>
      <span class="badge">🚕 Dự đoán thời gian chuyến taxi</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 1. NẠP MÔ HÌNH
# =========================================================

@st.cache_resource
def get_model(model_name):
    return load_selected_model(model_name)


st.subheader("01 · Chọn cách dự đoán")

options = list(AVAILABLE_MODELS) + ["So sánh tất cả mô hình"]

selected_model = st.selectbox(
    "Mô hình dự đoán",
    options,
    index=0,
)

if selected_model == "So sánh tất cả mô hình":
    st.caption("Đặt kết quả LightGBM, Ridge và Random Forest cạnh nhau để so sánh.")
else:
    descriptions = {
        "LightGBM": "Mô hình cuối được huấn luyện trên toàn bộ dữ liệu Train với cấu hình đã chọn sau tối ưu.",
        "Ridge": "Mô hình hồi quy tuyến tính có điều chuẩn, dùng làm đối chiếu.",
        "RandomForest": "Mô hình rừng ngẫu nhiên, kết hợp nhiều cây quyết định.",
    }
    st.caption(descriptions[selected_model])


# =========================================================
# 2. NHẬP THÔNG TIN CHUYẾN ĐI
# =========================================================

st.subheader("02 · Thiết lập chuyến đi")
st.caption("Các giá trị có sẵn là một chuyến đi mẫu. Thay đổi theo dữ liệu của bạn.")

with st.form("trip_form"):

    st.markdown("#### Thời gian & hành khách")
    left, right = st.columns(2)

    vendor = left.number_input(
        "Mã nhà vận tải",
        min_value=1,
        max_value=100,
        value=1,
    )

    passengers = right.number_input(
        "Số hành khách",
        min_value=0,
        max_value=20,
        value=1,
    )

    month = left.selectbox(
        "Tháng đón",
        list(range(1, 13)),
        index=2,
    )

    dow = right.selectbox(
        "Ngày trong tuần",
        list(range(7)),
        index=2,
        format_func=lambda value: ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"][value],
    )

    hour = left.slider(
        "Giờ đón",
        0, 23, 8,
    )

    minute = right.selectbox(
        "Mốc phút",
        [0, 15, 30, 45],
    )

    st.divider()
    st.markdown("#### Điểm đón & điểm đến")
    st.caption("Nhập tọa độ thập phân. Khoảng cách bên dưới là đường chim bay, tính bằng km.")
    left, right = st.columns(2)
    left.markdown("**🟢 Điểm đón**")
    right.markdown("**📍 Điểm đến**")
    pickup_lon = left.number_input(
        "Kinh độ điểm đón",
        value=-73.98,
        min_value=-180.0,
        max_value=180.0,
        format="%.4f",
    )

    pickup_lat = left.number_input(
        "Vĩ độ điểm đón",
        value=40.75,
        min_value=-90.0,
        max_value=90.0,
        format="%.4f",
    )

    dropoff_lon = right.number_input(
        "Kinh độ điểm đến",
        value=-73.96,
        min_value=-180.0,
        max_value=180.0,
        format="%.4f",
    )

    dropoff_lat = right.number_input(
        "Vĩ độ điểm đến",
        value=40.77,
        min_value=-90.0,
        max_value=90.0,
        format="%.4f",
    )

    distance = left.number_input(
        "Khoảng cách đường chim bay (km)",
        min_value=0.0,
        value=2.5,
        step=0.5,
    )

    flag = right.selectbox(
        "Cờ lưu/đồng bộ dữ liệu",
        ["N", "Y"],
        format_func=lambda value: "N · Không lưu tạm" if value == "N" else "Y · Có lưu tạm",
        help="Giữ nguyên giá trị N/Y trong dữ liệu chuyến đi.",
    )

    submitted = st.form_submit_button(
        "Ước tính thời gian chuyến đi →",
        type="primary",
        width="stretch",
    )


# =========================================================
# 3. TẠO DỮ LIỆU ĐẦU VÀO
# =========================================================

if submitted:

    row = pd.DataFrame([{
        "vendor_id": vendor,
        "passenger_count": passengers,
        "pickup_month": month,
        "pickup_dayofweek": dow,
        "pickup_hour": hour,
        "pickup_minute_bucket": minute,
        "pickup_lon_approx": pickup_lon,
        "pickup_lat_approx": pickup_lat,
        "dropoff_lon_approx": dropoff_lon,
        "dropoff_lat_approx": dropoff_lat,
        "straight_line_km_approx": distance,
        "store_and_fwd_flag": flag,
    }])

    numeric_values = row.select_dtypes("number").to_numpy()

    if not np.isfinite(numeric_values).all():
        st.error("Dữ liệu đầu vào không hợp lệ.")
        st.stop()

    # -----------------------------------------------------
    # 4. DỰ ĐOÁN
    # -----------------------------------------------------

    st.subheader("03 · Thời gian ước tính")
    st.caption(f"Chuyến đi {distance:g} km · {hour:02d}:{minute:02d} · {passengers} hành khách")

    if selected_model == "So sánh tất cả mô hình":
        model_names = list(AVAILABLE_MODELS)
    else:
        model_names = [selected_model]

    results = []

    for name in model_names:
        try:
            loaded = get_model(name)

            seconds = float(
                predict_selected_model(
                    row,
                    name,
                    loaded,
                )[0]
            )

            results.append({
                "Mô hình": name,
                "Thời gian (giây)": round(seconds, 1),
                "Thời gian (phút)": round(seconds / 60, 2),
            })

        except Exception as exc:
            st.error(f"Lỗi mô hình {name}: {exc}")

    if results:
        result_df = pd.DataFrame(results)

        cards = st.columns(len(results))
        for card, result in zip(cards, results):
            with card:
                st.metric(result["Mô hình"], f"{result['Thời gian (phút)']:.2f} phút")
                st.caption(f"Tương đương {result['Thời gian (giây)']:.1f} giây")

        if len(results) > 1:
            st.markdown("#### So sánh dự đoán")
            chart_tab, table_tab = st.tabs(["Biểu đồ", "Bảng chi tiết"])
            with table_tab:
                st.dataframe(result_df, hide_index=True, width="stretch")
            with chart_tab:
                st.bar_chart(
                    result_df.set_index("Mô hình")[["Thời gian (phút)"]],
                    y_label="Phút", color="#269c8c", height=280,
                )

        st.caption(
            "Thời gian thực tế còn phụ thuộc vào "
            "giao thông, lộ trình và điều kiện di chuyển."
        )
else:
    st.info("Điền thông tin và nhấn “Ước tính thời gian chuyến đi” để xem kết quả.")

st.divider()
st.caption("UNETI01 · Ước tính thời gian taxi bằng học máy")
