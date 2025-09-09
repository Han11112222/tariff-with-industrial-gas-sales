# app.py — ε(가동률→도시가스) 추정기 (정상 동작 버전)

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="ε(가동률→도시가스) 추정기", layout="wide")
st.title("ε(가동률→도시가스) 추정기")
st.caption("대구 산업용 도시가스(TJ)와 제조업 평균가동률(대구/전국)의 탄력치 ε 자동 회귀")

# ======================
# 사이드바
# ======================
with st.sidebar:
    st.header("입력")
    st.write("CSV 칼럼 필요: `date(YYYY-MM)`, `gas_TJ`, `util_daegu`, `util_korea`")
    uploaded = st.file_uploader("input_data.csv 업로드", type=["csv"])

    # 샘플 템플릿 다운로드
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
    util_source = st.selectbox("가동률 소스", ["daegu", "korea"], index=0)
    lags = st.multiselect("래그(月)", [0, 1, 2], default=[0, 1, 2])
    hac_lags = st.number_input("HAC(Newey-West) maxlags", min_value=0, max_value=24, value=6, step=1)
    run = st.button("회귀 실행")

# ======================
# 데이터 로딩 유틸
# ======================
# 업로드가 없고 리포 루트에 input_data.csv가 있으면 자동 사용
default_csv = Path("input_data.csv")
if uploaded is None and default_csv.exists():
    st.info("리포지토리의 input_data.csv를 자동으로 불러왔어.")
    uploaded = str(default_csv)  # 경로 문자열로 치환

@st.cache_data
def load_csv(file_like_or_path):
    p = Path(str(file_like_or_path))
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    else:
        return pd.read_csv(file_like_or_path, parse_dates=["date"])

def prep(df: pd.DataFrame) -> pd.DataFrame:
    req = {"date","gas_TJ","util_daegu","util_korea"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"필수 칼럼 누락: {missing}")
    df = df.sort_values("date").reset_index(drop=True).copy()
    for c in ["gas_TJ","util_daegu","util_korea"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["gas_TJ","util_daegu","util_korea"])
    EPS = 1e-6
    df["util_daegu_ratio"] = (df["util_daegu"].clip(EPS, 100-EPS)) / 100.0
    df["util_korea_ratio"] = (df["util_korea"].clip(EPS, 100-EPS)) / 100.0
    df["ln_gas"] = np.log(df["gas_TJ"].clip(lower=EPS))
    df["ln_util_daegu"] = np.log(df["util_daegu_ratio"])
    df["ln_util_korea"] = np.log(df["util_korea_ratio"])
    return df

def fit_ols(y, X, hac_lags=6):
    X_ = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model

def run_reg(df: pd.DataFrame, util_source: str, lags, hac_lags: int):
    recs = []
    models = {}
    for lag in sorted(set(lags)):
        util_col = f"util_{util_source}"
        ln_util_col = f"ln_util_{util_source}"
        if lag > 0:
            df[f"{util_col}_lag{lag}"] = df[util_col].shift(lag)
            df[f"{ln_util_col}_lag{lag}"] = df[f"ln_util_{util_source}"].shift(lag)
        else:
            df[f"{util_col}_lag{lag}"] = df[util_col]
            df[f"{ln_util_col}_lag{lag}"] = df[f"ln_util_{util_source}"]

        # A) log-log
        yA = df["ln_gas"]
        XA = df[[f"{ln_util_col}_lag{lag}"]]
        mA = fit_ols(yA, XA, hac_lags=hac_lags)
        models[("loglog", lag)] = mA
        recs.append({
            "model": "loglog",
            "lag_months": lag,
            "coef_main": mA.params.get(f"{ln_util_col}_lag{lag}", np.nan),  # ε
            "std_err": mA.bse.get(f"{ln_util_col}_lag{lag}", np.nan),
            "t_stat": mA.tvalues.get(f"{ln_util_col}_lag{lag}", np.nan),
            "p_value": mA.pvalues.get(f"{ln_util_col}_lag{lag}", np.nan),
            "adj_R2": mA.rsquared_adj, "AIC": mA.aic, "BIC": mA.bic, "nobs": int(mA.nobs)
        })

        # B) level-level
        yB = df["gas_TJ"]
        XB = df[[f"{util_col}_lag{lag}"]]
        mB = fit_ols(yB, XB, hac_lags=hac_lags)
        models[("level", lag)] = mB
        recs.append({
            "model": "level",
            "lag_months": lag,
            "coef_main": mB.params.get(f"{util_col}_lag{lag}", np.nan),     # TJ per 1%-pt
            "std_err": mB.bse.get(f"{util_col}_lag{lag}", np.nan),
            "t_stat": mB.tvalues.get(f"{util_col}_lag{lag}", np.nan),
            "p_value": mB.pvalues.get(f"{util_col}_lag{lag}", np.nan),
            "adj_R2": mB.rsquared_adj, "AIC": mB.aic, "BIC": mB.bic, "nobs": int(mB.nobs)
        })

    res = pd.DataFrame.from_records(recs).sort_values(["model","AIC"]).reset_index(drop=True)

    # pick best among loglog
    best = res.query("model=='loglog'").sort_values("AIC").head(1)
    if best.empty:  # fallback
        best = res.sort_values("AIC").head(1)
    best_row = best.iloc[0].to_dict()
    best_model = models[(best_row["model"], int(best_row["lag_months"]))]

    return res, best_row, best_model

def gas_pct_from_pp(util_level_pct: float, delta_pp: float, elasticity: float) -> float:
    u0 = max(1e-6, min(99.999999, util_level_pct))/100.0
    u1 = max(1e-6, min(99.999999, util_level_pct + delta_pp))/100.0
    return elasticity * np.log(u1/u0) * 100.0

# ======================
# 실행
# ======================
if uploaded is not None and run:
    try:
        raw = load_csv(uploaded)
        df = prep(raw)
        res, best_row, best_model = run_reg(df, util_source, lags, hac_lags)

        st.subheader("모형 비교표")
        st.dataframe(res, use_container_width=True)
        st.download_button("결과 CSV 다운로드",
                           res.to_csv(index=False).encode("utf-8-sig"),
                           file_name="epsilon_results.csv", mime="text/csv")

        st.subheader("베스트 모형 요약")
        st.write(f"**Model**: {best_row['model']}  |  **Lag**: {int(best_row['lag_months'])}  "
                 f"|  **adj.R²**: {best_row['adj_R2']:.4f}  |  **AIC**: {best_row['AIC']:.2f}  "
                 f"|  **n**: {int(best_row['nobs'])}")
        st.write(f"**계수(coef_main)**: {best_row['coef_main']:.6f}  |  **p-value**: {best_row['p_value']:.4f}")

        with st.expander("statsmodels summary 보기"):
            st.text(best_model.summary().as_text())

        st.subheader("시나리오 계산기 (log-log 전용)")
        if best_row["model"] == "loglog":
            base_util = float(df[f"util_{util_source}"].dropna().iloc[-1])
            base_util = st.number_input("기준 가동률(%)", value=float(round(base_util, 2)), step=0.1)
            col1, col2, col3 = st.columns(3)
            with col1: dpp1 = st.number_input("Δpp #1", value=-1.0, step=0.1)
            with col2: dpp2 = st.number_input("Δpp #2", value=-1.5, step=0.1)
            with col3: dpp3 = st.number_input("Δpp #3", value=-2.0, step=0.1)
            eps = float(best_row["coef_main"])
            out = pd.DataFrame({
                "Δpp": [dpp1, dpp2, dpp3],
                "gas_%change": [gas_pct_from_pp(base_util, dpp1, eps),
                                gas_pct_from_pp(base_util, dpp2, eps),
                                gas_pct_from_pp(base_util, dpp3, eps)]
            }).round(3)
            st.table(out)
        else:
            st.info("베스트 모형이 level형이면 pp→% 변환 공식이 달라져 log-log 계산기를 숨겼어.")

    except Exception as e:
        st.error(f"실행 중 오류: {e}")

else:
    st.info("왼쪽에서 CSV 업로드 후 [회귀 실행]을 눌러줘. (리포에 input_data.csv가 있으면 자동으로 불러와)")
