
# epsilon_regression_template.py
# Purpose: Estimate elasticity (epsilon) between monthly industrial city-gas sales in Daegu and manufacturing utilization rates (Daegu/National).
# Inputs:  input_data.csv with columns [date (YYYY-MM), gas_TJ, util_daegu, util_korea]
# Outputs: epsilon_results.csv (model comparison), best_model.txt (detailed summary), example_scenario.txt (optional)
#
# Notes:
# - Use log-log model: ln(gas) ~ ln(util). Utilization is in percent 0-100; convert to ratio 0-1 before log.
# - Use HAC (Newey-West) robust standard errors to handle autocorrelation.
# - Try lags of utilization: 0, 1, 2 months.

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

INPUT = Path("input_data.csv")
df = pd.read_csv(INPUT, parse_dates=["date"])

required = {"date","gas_TJ","util_daegu","util_korea"}
if not required.issubset(df.columns):
    missing = required - set(df.columns)
    raise ValueError(f"Missing columns: {missing}")

df = df.sort_values("date").reset_index(drop=True)
for c in ["gas_TJ","util_daegu","util_korea"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["gas_TJ","util_daegu","util_korea"])

EPS = 1e-6
df["util_daegu_ratio"] = (df["util_daegu"].clip(EPS, 100-EPS)) / 100.0
df["util_korea_ratio"] = (df["util_korea"].clip(EPS, 100-EPS)) / 100.0
df["ln_gas"] = np.log(df["gas_TJ"].clip(lower=EPS))
df["ln_util_daegu"] = np.log(df["util_daegu_ratio"])
df["ln_util_korea"] = np.log(df["util_korea_ratio"])

def fit_ols(y, X, hac_lags=6):
    X_ = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model

records = []
models_store = {}
for src in ["daegu","korea"]:
    for lag in [0,1,2]:
        util_col = f"util_{src}"
        ln_util_col = f"ln_util_{src}"
        if lag > 0:
            df[f"{util_col}_lag{lag}"] = df[util_col].shift(lag)
            df[f"{ln_util_col}_lag{lag}"] = df[f"ln_util_{src}"].shift(lag)
        else:
            df[f"{util_col}_lag{lag}"] = df[util_col]
            df[f"{ln_util_col}_lag{lag}"] = df[f"ln_util_{src}"]

        # Model A: log-log
        yA = df["ln_gas"]
        XA = df[[f"{ln_util_col}_lag{lag}"]]
        mA = fit_ols(yA, XA)
        models_store[("loglog",src,lag)] = mA
        coefA = mA.params.get(f"{ln_util_col}_lag{lag}", np.nan)
        seA   = mA.bse.get(f"{ln_util_col}_lag{lag}", np.nan)
        tA    = mA.tvalues.get(f"{ln_util_col}_lag{lag}", np.nan)
        pA    = mA.pvalues.get(f"{ln_util_col}_lag{lag}", np.nan)

        records.append({
            "model": "loglog",
            "util_source": src,
            "lag_months": lag,
            "coef_main": coefA,  # elasticity epsilon
            "std_err": seA,
            "t_stat": tA,
            "p_value": pA,
            "adj_R2": mA.rsquared_adj,
            "AIC": mA.aic,
            "BIC": mA.bic,
            "nobs": int(mA.nobs)
        })

        # Model B: level-level
        yB = df["gas_TJ"]
        XB = df[[f"{util_col}_lag{lag}"]]
        mB = fit_ols(yB, XB)
        models_store[("level",src,lag)] = mB
        coefB = mB.params.get(f"{util_col}_lag{lag}", np.nan)
        seB   = mB.bse.get(f"{util_col}_lag{lag}", np.nan)
        tB    = mB.tvalues.get(f"{util_col}_lag{lag}", np.nan)
        pB    = mB.pvalues.get(f"{util_col}_lag{lag}", np.nan)

        records.append({
            "model": "level",
            "util_source": src,
            "lag_months": lag,
            "coef_main": coefB,  # TJ per 1%-point of utilization
            "std_err": seB,
            "t_stat": tB,
            "p_value": pB,
            "adj_R2": mB.rsquared_adj,
            "AIC": mB.aic,
            "BIC": mB.bic,
            "nobs": int(mB.nobs)
        })

res = pd.DataFrame.from_records(records).sort_values(["model","AIC"]).reset_index(drop=True)
res.to_csv("epsilon_results.csv", index=False, encoding="utf-8-sig")

# pick best among log-log by AIC; fallback to global best by AIC
best = res.query("model=='loglog'").sort_values("AIC").head(1)
if best.empty:
    best = res.sort_values("AIC").head(1)
best_info = best.iloc[0].to_dict()
best_key = (best_info["model"], best_info["util_source"], int(best_info["lag_months"]))
best_model = models_store[best_key]

with open("best_model.txt","w", encoding="utf-8") as f:
    f.write("=== Best model summary ===\n")
    f.write(f"Model type : {best_info['model']}\n")
    f.write(f"Util source: {best_info['util_source']}\n")
    f.write(f"Lag (m)   : {best_info['lag_months']}\n")
    f.write(f"AIC       : {best_info['AIC']:.3f}\n")
    f.write(f"adj.R^2   : {best_info['adj_R2']:.4f}\n")
    f.write(f"Coef(main): {best_info['coef_main']:.6f}\n")
    f.write("\n--- statsmodels summary ---\n")
    f.write(str(best_model.summary()))
    f.write("\n")
    f.write("\nUsage notes:\n")
    f.write(" * If model=='loglog', coef_main is elasticity (epsilon).\n")
    f.write(" * 1% relative change in utilization implies approx epsilon% change in gas.\n")
    f.write(" * To convert percent-point (pp) changes to percent changes, use helper below.\n")

# helper: convert pp change in utilization to percent change in gas (log-log model)
def gas_pct_change_from_pp(util_level_pct: float, delta_pp: float, elasticity: float) -> float:
    u0 = max(1e-6, min(99.999999, util_level_pct))/100.0
    u1 = max(1e-6, min(99.999999, util_level_pct + delta_pp))/100.0
    return elasticity * np.log(u1/u0) * 100.0  # return percent

# optional example
try:
    if best_info["model"]=="loglog":
        eps = best_info["coef_main"]
        base_u = float(df["util_daegu"].dropna().iloc[-1])
        dpp = -1.0
        ex_pct = gas_pct_change_from_pp(base_u, dpp, eps)
        with open("example_scenario.txt","w", encoding="utf-8") as f:
            f.write(f"Base util {base_u:.2f}% with {dpp:+.1f}pp change => gas {ex_pct:.3f}% change (epsilon={eps:.4f})\n")
except Exception:
    pass
