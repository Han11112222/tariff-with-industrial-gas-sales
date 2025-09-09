with st.sidebar:
    st.header("입력")
    st.write("CSV 형식: date(YYYY-MM), gas_TJ, util_daegu, util_korea")
    uploaded = st.file_uploader("input_data.csv 업로드", type=["csv"])

    # 샘플 템플릿 다운로드
    import pandas as pd
    _sample = pd.DataFrame({
        "date": ["2023-01","2023-02","2023-03"],
        "gas_TJ": [None, None, None],
        "util_daegu": [None, None, None],
        "util_korea": [None, None, None],
    })
    _csv = _sample.to_csv(index=False).encode("utf-8-sig")
    st.download_button("샘플 템플릿 CSV 다운로드", _csv,
                       file_name="input_data_template.csv", mime="text/csv")

    st.divider()
    st.header("옵션")
    # (여기에 기존 selectbox/lag 입력 등 옵션 그대로)
