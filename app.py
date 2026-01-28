# app.py
# -*- coding: utf-8 -*-
import io
import os
import re
import math
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import pearsonr
from analysis import run_item_analysis, normalize_item_columns


# ---- Optional GPT report (if gpt_report.py exists & has generate_gpt_report) ----
GPT_AVAILABLE = False
generate_gpt_report = None
try:
    from gpt_report import generate_gpt_report  # type: ignore
    GPT_AVAILABLE = callable(generate_gpt_report)
except Exception:
    GPT_AVAILABLE = False
    generate_gpt_report = None


# ---- Page ----
st.set_page_config(page_title="Scale Item Analysis MVP", layout="wide")
st.title("📊 Scale Item Analysis MVP")


# ---- Helpers ----
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV loader for Streamlit UploadedFile.
    Tries common encodings and handles BOM.
    """
    if uploaded_file is None:
        raise ValueError("尚未上傳 CSV 檔案。")

    raw = uploaded_file.getvalue()
    if raw is None or len(raw) == 0:
        raise ValueError("上傳的檔案是空的（0 bytes）。請確認 CSV 內容是否存在。")

    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            bio = io.BytesIO(raw)
            return pd.read_csv(bio, encoding=enc)
        except Exception as e:
            last_err = e

    raise ValueError(f"讀取 CSV 失敗（已嘗試 {encodings}）。最後錯誤：{repr(last_err)}")


def safe_show_exception(e: Exception):
    st.error("發生錯誤（safe）")
    st.code(repr(e))
    with st.expander("Traceback（除錯用）"):
        st.code(traceback.format_exc())


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Excel-friendly: UTF-8 with BOM
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ===== Item code detection =====
ITEM_CODE_RE = re.compile(r"^[A-Za-z]\d{2,3}(_\d+)?$")


def _find_item_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        s = str(c).strip()
        if ITEM_CODE_RE.match(s):
            cols.append(s)
    return cols


def _dim_letter(code: str) -> str | None:
    m = re.match(r"^([A-Za-z])", str(code))
    return m.group(1).upper() if m else None


def build_dim_means_per_row(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    產生逐列（每份問卷一列）的構面平均：
    - 依題項代碼第一碼決定構面（A/B/C...）
    - 每列對該構面所有題目做 mean(axis=1, skipna=True)
    - 輸出為「4 位小數字串」，未滿補 0（例如 3.5 → 3.5000）
    """
    item_cols_all = _find_item_cols(df_norm)
    if not item_cols_all:
        return pd.DataFrame()

    dims = sorted({d for d in (_dim_letter(c) for c in item_cols_all) if d is not None})

    df_item = df_norm[item_cols_all].apply(pd.to_numeric, errors="coerce")

    out = pd.DataFrame(index=df_norm.index)
    for d in dims:
        cols_d = [c for c in item_cols_all if _dim_letter(c) == d]
        mean_series = df_item[cols_d].mean(axis=1, skipna=True)
        out[d] = mean_series.apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    return out


# ===== Regression table =====
def _sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_regression_table(df: pd.DataFrame, iv_vars: list[str], dv_var: str):
    """
    產生迴歸表（比照論文表格）：
    - 未標準化係數（b；欄名仍用「β估計值」以符合你的表頭）
    - 標準化係數 Beta（Beta = b * sd(x) / sd(y)）
    - t、顯著性(p)
    - F、P(F)、R²、Adj R²、N
    """
    if not iv_vars or not dv_var:
        raise ValueError("請先設定自變數與依變數。")

    cols = iv_vars + [dv_var]
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if d.empty:
        raise ValueError("可用資料為空（IV/DV 可能有空值或非數值）。")

    y = d[dv_var].astype(float)
    X = d[iv_vars].astype(float)
    Xc = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, Xc).fit()

    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues

    sd_y = y.std(ddof=1)
    sd_x = X.std(ddof=1)

    beta_std = {}
    for v in iv_vars:
        if sd_y == 0 or pd.isna(sd_y) or sd_x[v] == 0 or pd.isna(sd_x[v]):
            beta_std[v] = np.nan
        else:
            beta_std[v] = params[v] * (sd_x[v] / sd_y)

    rows = []
    rows.append(
        {
            "自變項": "（常數）",
            "未標準化係數 β估計值": f"{params['const']:.3f}",
            "標準化係數 Beta": "—",
            "t": f"{tvals['const']:.3f}{_sig_stars(pvals['const'])}",
            "顯著性": f"{pvals['const']:.3f}",
        }
    )

    for v in iv_vars:
        rows.append(
            {
                "自變項": v,
                "未標準化係數 β估計值": f"{params[v]:.3f}",
                "標準化係數 Beta": ("" if pd.isna(beta_std[v]) else f"{beta_std[v]:.3f}"),
                "t": f"{tvals[v]:.3f}{_sig_stars(pvals[v])}",
                "顯著性": f"{pvals[v]:.3f}",
            }
        )

    table_df = pd.DataFrame(rows)

    summary = {
        "F": float(model.fvalue) if model.fvalue is not None else np.nan,
        "P(F)": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "R2": float(model.rsquared),
        "Adj_R2": float(model.rsquared_adj),
        "N": int(model.nobs),
    }
    return table_df, summary


# ===== Mediation analysis (IV -> M -> DV) =====
def _to_num_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")


def _fit_ols(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit()


def build_mediation_results(
    df: pd.DataFrame,
    iv: str,
    med: str,
    dv: str,
    n_boot: int = 2000,
    seed: int = 42,
):
    """
    產出中介分析（OLS）：
    - 路徑 a: M ~ IV
    - 路徑 c: DV ~ IV
    - 路徑 b & c': DV ~ IV + M
    - indirect = a*b
    - Sobel z / p（近似）
    - bootstrap CI（percentile）
    """
    d = _to_num_df(df, [iv, med, dv])
    if d.empty:
        raise ValueError("可用資料為空（IV/M/DV 可能有空值或非數值）。")

    # a path
    m_a = _fit_ols(d[med], d[[iv]])
    a = float(m_a.params[iv])
    se_a = float(m_a.bse[iv])
    p_a = float(m_a.pvalues[iv])

    # c path (total)
    m_c = _fit_ols(d[dv], d[[iv]])
    c = float(m_c.params[iv])
    se_c = float(m_c.bse[iv])
    p_c = float(m_c.pvalues[iv])

    # b and c' path
    m_bc = _fit_ols(d[dv], d[[iv, med]])
    b = float(m_bc.params[med])
    se_b = float(m_bc.bse[med])
    p_b = float(m_bc.pvalues[med])

    c_prime = float(m_bc.params[iv])
    se_cprime = float(m_bc.bse[iv])
    p_cprime = float(m_bc.pvalues[iv])

    indirect = a * b

    
    # Sobel test (normal approximation)
    sobel_se = math.sqrt((b * b * se_a * se_a) + (a * a * se_b * se_b))
    sobel_z = (indirect / sobel_se) if sobel_se != 0 else float("nan")

    if np.isfinite(sobel_z):
        sobel_p = float(2 * (1 - norm.cdf(abs(sobel_z))))
    else:
        sobel_p = float("nan")

    # Bootstrap CI for indirect
    rng = np.random.default_rng(seed)
    n = len(d)
    inds = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        ds = d.iloc[idx]
        try:
            ma = _fit_ols(ds[med], ds[[iv]])
            mbc = _fit_ols(ds[dv], ds[[iv, med]])
            inds.append(float(ma.params[iv]) * float(mbc.params[med]))
        except Exception:
            continue

    if len(inds) >= 20:
        ci_low, ci_high = np.percentile(inds, [2.5, 97.5])
    else:
        ci_low, ci_high = (np.nan, np.nan)

    paths_df = pd.DataFrame(
        [
            {"路徑": "a (IV→M)", "係數": a, "SE": se_a, "t": float(m_a.tvalues[iv]), "p": p_a},
            {"路徑": "c (IV→DV total)", "係數": c, "SE": se_c, "t": float(m_c.tvalues[iv]), "p": p_c},
            {"路徑": "b (M→DV | IV)", "係數": b, "SE": se_b, "t": float(m_bc.tvalues[med]), "p": p_b},
            {"路徑": "c' (IV→DV direct | M)", "係數": c_prime, "SE": se_cprime, "t": float(m_bc.tvalues[iv]), "p": p_cprime},
        ]
    )

    effects_df = pd.DataFrame(
        [
            {
                "效果": "Indirect (a*b)",
                "值": indirect,
                "Sobel z": sobel_z,
                "Sobel p": sobel_p,
                "Boot CI 2.5%": ci_low,
                "Boot CI 97.5%": ci_high,
            }
        ]
    )

    summary = {
        "N": int(n),
        "indirect": float(indirect),
        "sobel_z": float(sobel_z) if np.isfinite(sobel_z) else np.nan,
        "sobel_p": float(sobel_p) if np.isfinite(sobel_p) else np.nan,
        "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "boot_used": int(len(inds)),
    }
    return paths_df, effects_df, summary

from statsmodels.stats.stattools import durbin_watson

def _std_beta(params: pd.Series, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    計算標準化係數 Beta：Beta = b * sd(x) / sd(y)
    """
    sd_y = y.std(ddof=1)
    sd_x = X.std(ddof=1)
    out = {}
    for v in X.columns:
        if sd_y == 0 or pd.isna(sd_y) or sd_x[v] == 0 or pd.isna(sd_x[v]):
            out[v] = np.nan
        else:
            out[v] = float(params[v]) * float(sd_x[v] / sd_y)
    return out


def _fmt_beta(beta: float, p: float) -> str:
    if pd.isna(beta):
        return ""
    stars = _sig_stars(p)
    return f"{beta:.3f}{stars}"


def _fmt_t(t: float) -> str:
    if pd.isna(t):
        return ""
    return f"{t:.3f}"


def build_mediation_paper_table(df: pd.DataFrame, iv: str, med: str, dv: str):
    """
    產出論文式中介分析迴歸表（對應你右邊那張表）：

    條件二：DV=med, IV=[iv]
    條件一：DV=dv,  IV=[iv]
    條件三：DV=dv,  IV=[iv, med]

    輸出欄位：
    - 每個條件：β值（標準化係數）與 t 值
    - R²、ΔR²(=Adj R²)、F、D-W
    """

    d = df[[iv, med, dv]].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if d.empty:
        raise ValueError("可用資料為空（IV/M/DV 可能有空值或非數值）。")

    # ---- Condition 2: M ~ IV ----
    y2 = d[med].astype(float)
    X2 = d[[iv]].astype(float)
    m2 = _fit_ols(y2, X2)
    beta2 = _std_beta(m2.params, X2, y2)

    # ---- Condition 1: DV ~ IV ----
    y1 = d[dv].astype(float)
    X1 = d[[iv]].astype(float)
    m1 = _fit_ols(y1, X1)
    beta1 = _std_beta(m1.params, X1, y1)

    # ---- Condition 3: DV ~ IV + M ----
    y3 = d[dv].astype(float)
    X3 = d[[iv, med]].astype(float)
    m3 = _fit_ols(y3, X3)
    beta3 = _std_beta(m3.params, X3, y3)

    # 欄位名（對應你的表頭替換）
    col_c2_beta = f"{med}（條件二）β值"
    col_c2_t    = f"{med}（條件二）t值"
    col_c1_beta = f"{dv}（條件一）β值"
    col_c1_t    = f"{dv}（條件一）t值"
    col_c3_beta = f"{dv}（條件三）β值"
    col_c3_t    = f"{dv}（條件三）t值"

    # 表格列：IV, M, R², ΔR²(Adj R²), F, D-W
    rows = []

    # 自變項（IV）列
    rows.append({
        "自變項": iv,
        col_c2_beta: _fmt_beta(beta2.get(iv, np.nan), float(m2.pvalues.get(iv, np.nan))),
        col_c2_t:    _fmt_t(float(m2.tvalues.get(iv, np.nan))),
        col_c1_beta: _fmt_beta(beta1.get(iv, np.nan), float(m1.pvalues.get(iv, np.nan))),
        col_c1_t:    _fmt_t(float(m1.tvalues.get(iv, np.nan))),
        col_c3_beta: _fmt_beta(beta3.get(iv, np.nan), float(m3.pvalues.get(iv, np.nan))),
        col_c3_t:    _fmt_t(float(m3.tvalues.get(iv, np.nan))),
    })

    # 中介變項（M）列（只有條件三有）
    rows.append({
        "自變項": med,
        col_c2_beta: "",
        col_c2_t:    "",
        col_c1_beta: "",
        col_c1_t:    "",
        col_c3_beta: _fmt_beta(beta3.get(med, np.nan), float(m3.pvalues.get(med, np.nan))),
        col_c3_t:    _fmt_t(float(m3.tvalues.get(med, np.nan))),
    })

    # R²
    rows.append({
        "自變項": "R²",
        col_c2_beta: f"{float(m2.rsquared):.3f}",
        col_c2_t:    "",
        col_c1_beta: f"{float(m1.rsquared):.3f}",
        col_c1_t:    "",
        col_c3_beta: f"{float(m3.rsquared):.3f}",
        col_c3_t:    "",
    })

    # ΔR²（你右邊表其實是 Adj R²，數字差很小：0.576 vs 0.575 那種）
    rows.append({
        "自變項": "ΔR²",
        col_c2_beta: f"{float(m2.rsquared_adj):.3f}",
        col_c2_t:    "",
        col_c1_beta: f"{float(m1.rsquared_adj):.3f}",
        col_c1_t:    "",
        col_c3_beta: f"{float(m3.rsquared_adj):.3f}",
        col_c3_t:    "",
    })

    # F
    rows.append({
        "自變項": "F",
        col_c2_beta: f"{float(m2.fvalue):.3f}{_sig_stars(float(m2.f_pvalue))}",
        col_c2_t:    "",
        col_c1_beta: f"{float(m1.fvalue):.3f}{_sig_stars(float(m1.f_pvalue))}",
        col_c1_t:    "",
        col_c3_beta: f"{float(m3.fvalue):.3f}{_sig_stars(float(m3.f_pvalue))}",
        col_c3_t:    "",
    })

    # D-W
    rows.append({
        "自變項": "D-W",
        col_c2_beta: f"{float(durbin_watson(m2.resid)):.3f}",
        col_c2_t:    "",
        col_c1_beta: f"{float(durbin_watson(m1.resid)):.3f}",
        col_c1_t:    "",
        col_c3_beta: f"{float(durbin_watson(m3.resid)):.3f}",
        col_c3_t:    "",
    })

    table_df = pd.DataFrame(rows)

    meta = {
        "N": int(m3.nobs),
        "cond1": m1,
        "cond2": m2,
        "cond3": m3,
    }
    return table_df, meta


def build_moderation_paper_table(df: pd.DataFrame, iv: str, mod: str, dv: str):
    """
    產出論文式干擾分析迴歸表（對應你右邊那張表：模型一/二/三）

    模型一：DV ~ IV
    模型二：DV ~ IV + MOD
    模型三：DV ~ IV + MOD + (IV×MOD)

    輸出欄位：
    - 每個模型：β值（標準化係數）與 t 值
    - R²、ΔR²（這裡是「R² change」，對應你圖的 0.063/0.001 那種）
    - F
    """

    d = df[[iv, mod, dv]].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if d.empty:
        raise ValueError("可用資料為空（IV/MOD/DV 可能有空值或非數值）。")

    # interaction term（不做中心化，完全照你圖的做法；若你要中心化我可再加 toggle）
    inter_name = f"{iv}×{mod}"
    d[inter_name] = d[iv] * d[mod]

    # ---- Model 1: DV ~ IV ----
    y1 = d[dv].astype(float)
    X1 = d[[iv]].astype(float)
    m1 = _fit_ols(y1, X1)
    beta1 = _std_beta(m1.params, X1, y1)

    # ---- Model 2: DV ~ IV + MOD ----
    y2 = d[dv].astype(float)
    X2 = d[[iv, mod]].astype(float)
    m2 = _fit_ols(y2, X2)
    beta2 = _std_beta(m2.params, X2, y2)

    # ---- Model 3: DV ~ IV + MOD + IV×MOD ----
    y3 = d[dv].astype(float)
    X3 = d[[iv, mod, inter_name]].astype(float)
    m3 = _fit_ols(y3, X3)
    beta3 = _std_beta(m3.params, X3, y3)

    # 表頭（對齊你圖）
    col_m1_beta = f"{dv}（模型一）β值"
    col_m1_t    = f"{dv}（模型一）t值"
    col_m2_beta = f"{dv}（模型二）β值"
    col_m2_t    = f"{dv}（模型二）t值"
    col_m3_beta = f"{dv}（模型三）β值"
    col_m3_t    = f"{dv}（模型三）t值"

    # ΔR² = R² change（模型二-模型一；模型三-模型二；模型一留空或=R²都行）
    
    r2_1 = float(m1.rsquared)
    r2_2 = float(m2.rsquared)
    r2_3 = float(m3.rsquared)

    # ΔR² = R² change（嚴格定義）
    dr2_1 = np.nan                # 模型一不計 ΔR²（論文通常留空）
    dr2_2 = r2_2 - r2_1           # 模型二 − 模型一
    dr2_3 = r2_3 - r2_2           # 模型三 − 模型二

    rows = []

    # IV row
    rows.append({
        "自變項": iv,
        col_m1_beta: _fmt_beta(beta1.get(iv, np.nan), float(m1.pvalues.get(iv, np.nan))),
        col_m1_t:    _fmt_t(float(m1.tvalues.get(iv, np.nan))),
        col_m2_beta: _fmt_beta(beta2.get(iv, np.nan), float(m2.pvalues.get(iv, np.nan))),
        col_m2_t:    _fmt_t(float(m2.tvalues.get(iv, np.nan))),
        col_m3_beta: _fmt_beta(beta3.get(iv, np.nan), float(m3.pvalues.get(iv, np.nan))),
        col_m3_t:    _fmt_t(float(m3.tvalues.get(iv, np.nan))),
    })

    # MOD row
    rows.append({
        "自變項": mod,
        col_m1_beta: "",
        col_m1_t:    "",
        col_m2_beta: _fmt_beta(beta2.get(mod, np.nan), float(m2.pvalues.get(mod, np.nan))),
        col_m2_t:    _fmt_t(float(m2.tvalues.get(mod, np.nan))),
        col_m3_beta: _fmt_beta(beta3.get(mod, np.nan), float(m3.pvalues.get(mod, np.nan))),
        col_m3_t:    _fmt_t(float(m3.tvalues.get(mod, np.nan))),
    })

    # Interaction row (IV×MOD)
    rows.append({
        "自變項": f"{iv}*{mod}",
        col_m1_beta: "",
        col_m1_t:    "",
        col_m2_beta: "",
        col_m2_t:    "",
        col_m3_beta: _fmt_beta(beta3.get(inter_name, np.nan), float(m3.pvalues.get(inter_name, np.nan))),
        col_m3_t:    _fmt_t(float(m3.tvalues.get(inter_name, np.nan))),
    })

    # R² row
    rows.append({
        "自變項": "R²",
        col_m1_beta: f"{r2_1:.3f}",
        col_m1_t:    "",
        col_m2_beta: f"{r2_2:.3f}",
        col_m2_t:    "",
        col_m3_beta: f"{r2_3:.3f}",
        col_m3_t:    "",
    })

    # ΔR² row (R² change)
    rows.append({
    "自變項": "ΔR²",
        col_m1_beta: "",
        col_m1_t:    "",
        col_m2_beta: f"{dr2_2:.3f}",
        col_m2_t:    "",
        col_m3_beta: f"{dr2_3:.3f}",
        col_m3_t:    "",
    })


    # F row
    rows.append({
        "自變項": "F",
        col_m1_beta: f"{float(m1.fvalue):.3f}{_sig_stars(float(m1.f_pvalue))}",
        col_m1_t:    "",
        col_m2_beta: f"{float(m2.fvalue):.3f}{_sig_stars(float(m2.f_pvalue))}",
        col_m2_t:    "",
        col_m3_beta: f"{float(m3.fvalue):.3f}{_sig_stars(float(m3.f_pvalue))}",
        col_m3_t:    "",
    })

    table_df = pd.DataFrame(rows)

    meta = {
        "N": int(m3.nobs),
        "interaction_col": inter_name,
    }
    return table_df, meta

from scipy.stats import pearsonr

def build_discriminant_validity_table(df_norm: pd.DataFrame, item_df: pd.DataFrame):
    """
    區別效度分析表（Correlation Matrix + Cronbach's α on diagonal）

    - 列／欄：子構面（A1, A2, A3, B1, …）
    - 對角線：該子構面整體 Cronbach's α
    - 非對角線（左下）：子構面平均分數之 Pearson correlation
    - 右上三角：留空
    """

    # 1️⃣  從 item analysis 結果抓子構面與 alpha
    sub_alpha = (
        item_df
        .groupby("子構面")["該子構面整體 α"]
        .first()
        .dropna()
        .to_dict()
    )

    sub_dims = sorted(sub_alpha.keys())  # A1, A2, A3, ...

    # 2️⃣ 建立每個子構面的「平均分數」
    sub_scores = {}
    for sd in sub_dims:
        cols = [
            c for c in df_norm.columns
            if isinstance(c, str) and c.startswith(sd)
        ]
        if cols:
            sub_scores[sd] = (
                df_norm[cols]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1)
            )

    score_df = pd.DataFrame(sub_scores).dropna(axis=0, how="any")

    # 3️⃣ 建立空白表格
    mat = pd.DataFrame("", index=sub_dims, columns=sub_dims)

    # 4️⃣ 填值
    for i, r in enumerate(sub_dims):
        for j, c in enumerate(sub_dims):
            if i == j:
                # 對角線：Cronbach's α
                try:
                    mat.loc[r, c] = f"{float(sub_alpha[r]):.3f}"
                except Exception:
                    mat.loc[r, c] = str(sub_alpha[r])
            elif i > j:
                # 左下三角：Pearson r
                r_val, p_val = pearsonr(score_df[r], score_df[c])
                star = "**" if p_val < 0.01 else ""
                mat.loc[r, c] = f"{r_val:.3f}{star}"
            else:
                # 右上三角：留空
                mat.loc[r, c] = ""

    return mat


# ---- Sidebar ----
with st.sidebar:
    st.header("設定")
    st.caption("1) 上傳 CSV → 2) 產出 Item Analysis → 3) 下載結果（CSV）")

    uploaded_file = st.file_uploader("上傳 CSV", type=["csv"])

    st.divider()
    st.subheader("GPT 論文報告生成（可選）")

    gpt_on = st.toggle("啟用 GPT 報告", value=False, help="需要 OpenAI API Key 與可用額度（quota）。")

    model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
    model_pick = st.selectbox("選擇 GPT 模型", options=model_options, index=0)
    model_custom = st.text_input("或自行輸入模型名稱（選填）", value="", placeholder="例如：gpt-4o-mini")
    model_name = (model_custom.strip() or model_pick).strip()

    api_key = st.text_input("OpenAI API Key（以 sk- 開頭）", type="password", value="")
    st.caption("建議用環境變數也可：先在系統設定 OPENAI_API_KEY，再留空此欄。")

    st.divider()
    st.subheader("子構面規則（你指定）")
    st.write("子構面只取題項代碼的**前兩碼**：例如 A01→A0、A11→A1、A105→A1")
    st.caption("※ 這個規則需由 analysis.py 的分群邏輯配合（若你已改好 analysis.py 就會生效）。")


# ---- Main ----
if uploaded_file is None:
    st.info("請先在左側上傳 CSV 檔案。")
    st.stop()

try:
    df_raw = read_csv_safely(uploaded_file)
except Exception as e:
    safe_show_exception(e)
    st.stop()

# 正規化欄名（支援 A01.題目 / A01 題目 / A01）
df_norm, mapping = normalize_item_columns(df_raw)

st.subheader("原始資料預覽（前 5 列）")
st.dataframe(df_raw.head(), width="stretch")

with st.expander("欄名正規化對照（原始欄名 → 題項代碼）"):
    if mapping:
        map_df = pd.DataFrame([{"原始欄名": k, "題項代碼": v} for k, v in mapping.items()])
        st.dataframe(map_df, width="stretch")
    else:
        st.write("未偵測到可正規化的題項欄名（請確認欄名格式）。")

# ---- Item Analysis ----
st.subheader("📈 Item Analysis 結果")

try:
    # =========================================================
    # 1️⃣ Item Analysis
    # =========================================================
    result_df = run_item_analysis(df_norm)
    st.success("Item analysis completed.")
    st.dataframe(result_df, width="stretch", height=520)

    st.download_button(
        "下載 Item Analysis 結果 CSV",
        data=df_to_csv_bytes(result_df),
        file_name="item_analysis_results.csv",
        mime="text/csv",
    )

    # =========================================================
    # 2️⃣ 構面逐列平均（僅供分析使用）
    # =========================================================
    df_dim_means_row = build_dim_means_per_row(df_norm)
    if df_dim_means_row.empty:
        st.warning("找不到題項代碼欄位，無法產生構面平均（A/B/C...）。")
        st.stop()

    df_raw_plus_dimmeans = df_norm.copy()
    for c in df_dim_means_row.columns:
        df_raw_plus_dimmeans[c] = df_dim_means_row[c]

    dim_cols = list(df_dim_means_row.columns)

    # =========================================================
    # 3️⃣ Discriminant Validity（獨立 try / except）
    # =========================================================
    st.divider()
    st.subheader("📊 區別效度分析表")

    try:
        disc_df = build_discriminant_validity_table(df_norm, result_df)

        st.dataframe(disc_df, width="stretch")
        st.caption(
            "註：對角線為各子構面之 Cronbach’s α；"
            "左下三角為子構面間之皮爾森相關係數（** P<0.01）。"
        )

        st.download_button(
            "下載 區別效度分析表 CSV",
            data=df_to_csv_bytes(disc_df),
            file_name="discriminant_validity_table.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("區別效度分析失敗（safe）")
        safe_show_exception(e)

    # =========================================================
    # 4️⃣ 研究變數設定（IV / DV）
    # =========================================================
    st.divider()
    st.subheader("📌 研究變數設定（自變數 / 依變數）")

    iv_vars = st.multiselect(
        "① 勾選自變數（可複選）",
        options=dim_cols,
        default=[],
    )

    dv_var = st.selectbox(
        "② 選擇依變數（單一）",
        options=[""] + dim_cols,
        index=0,
    )

    if dv_var and dv_var in iv_vars:
        st.error("⚠️ 依變數不可同時被選為自變數，請重新設定。")

    elif iv_vars and dv_var:
        st.success(f"研究模型：IV = {iv_vars} → DV = {dv_var}")

        df_research = df_raw_plus_dimmeans[iv_vars + [dv_var]].copy()
        st.dataframe(df_research, width="stretch")

        st.download_button(
            "下載 研究用資料 CSV（IV + DV）",
            data=df_to_csv_bytes(df_research),
            file_name="research_dataset_IV_DV.csv",
            mime="text/csv",
        )

        # =====================================================
        # 5️⃣ Regression
        # =====================================================
        st.divider()
        st.subheader("📊 迴歸分析表（論文格式）")

        if st.button("執行迴歸分析", type="primary"):
            try:
                reg_table, reg_sum = build_regression_table(
                    df_research, iv_vars, dv_var
                )

                st.dataframe(reg_table, width="stretch")
                st.markdown(
                    f"**F={reg_sum['F']:.3f}，P={reg_sum['P(F)']:.3f}，"
                    f"R²={reg_sum['R2']:.3f}，Adj R²={reg_sum['Adj_R2']:.3f}，"
                    f"N={reg_sum['N']}**"
                )

            except Exception as e:
                st.error("迴歸分析失敗（safe）")
                safe_show_exception(e)

    else:
        st.info("請先選擇至少一個自變數與一個依變數。")

except Exception as e:
    st.error("Item Analysis 主流程失敗（safe）")
    safe_show_exception(e)
    st.stop()

# ====== Mediation Settings (互斥：A/B/C/D... 只能出現在一個位置) ======
st.divider()
st.subheader("🧩 中介分析設定")

dim_cols_all = dim_cols  # A, B, C, D ...

col1, col2, col3 = st.columns(3)

with col1:
    iv_m = st.selectbox(
        "① 自變數（IV）",
        options=[""] + dim_cols_all,
        index=0,
        key="med_iv",
    )

with col2:
    med_options = [""] + [c for c in dim_cols_all if c != iv_m]
    med_m = st.selectbox(
        "② 中介變數（M）",
        options=med_options,
        index=0,
        key="med_m",
    )

with col3:
    dv_options = [""] + [c for c in dim_cols_all if c not in {iv_m, med_m}]
    dv_m = st.selectbox(
        "③ 依變數（DV）",
        options=dv_options,
        index=0,
        key="med_dv",
    )

chosen = [x for x in [iv_m, med_m, dv_m] if x]

if len(chosen) != len(set(chosen)):
    st.error("⚠️ IV / M / DV 不可重複，A、B、C、D… 每個只能出現在一個角色中。")

elif iv_m and med_m and dv_m:
    st.success(f"中介模型：{iv_m} → {med_m} → {dv_m}")

    st.markdown("### 研究用資料表（僅保留 IV / M / DV）")
    df_mediation = df_raw_plus_dimmeans[[iv_m, med_m, dv_m]].copy()
    st.dataframe(df_mediation, width="stretch")

    st.download_button(
        "下載 中介分析研究用資料 CSV（IV + M + DV）",
        data=df_to_csv_bytes(df_mediation),
        file_name=f"mediation_dataset_{iv_m}_{med_m}_{dv_m}.csv",
        mime="text/csv",
    )

    st.markdown("### 中介分析")

    n_boot = st.number_input(
        "Bootstrap 次數（建議 2000）",
        min_value=200,
        max_value=20000,
        value=2000,
        step=200,
    )

    if st.button("執行中介分析", type="primary", key="run_mediation"):
        try:
            paper_table, meta = build_mediation_paper_table(
                df_raw_plus_dimmeans,
                iv=iv_m,
                med=med_m,
                dv=dv_m,
            )

            st.markdown(
                f"### 中介變數（{med_m}）對 自變數（{iv_m}）與 依變數（{dv_m}）之中介分析表"
            )

            st.dataframe(paper_table, width="stretch")

            st.caption(
                "註：* P<0.05，** P<0.01，*** P<0.001；"
                "ΔR² 為調整後 R²（Adj R²）；D-W 為 Durbin–Watson。"
            )

            tag = f"{iv_m}_to_{med_m}_to_{dv_m}".replace(" ", "")
            st.download_button(
                "下載 中介分析表 CSV",
                data=df_to_csv_bytes(paper_table),
                file_name=f"mediation_table_{tag}.csv",
                mime="text/csv",
            )

            st.markdown(f"**N={meta['N']}**")

        except Exception as e:
            st.error("中介分析失敗（safe）")
            safe_show_exception(e)

else:
    st.info("請依序選擇 IV / M / DV（且三者不可重複）後，才會顯示中介分析資料與結果。")



# =========================
# Moderation (IV -> DV moderated by W)
# =========================
st.divider()
st.subheader("🧩 干擾分析設定")

col1, col2, col3 = st.columns(3)

with col1:
    iv_w = st.selectbox("① 自變數（IV）", options=[""] + dim_cols, index=0, key="mod_iv")

# moderator options exclude IV
mod_options = [""] + [c for c in dim_cols if c != iv_w]
with col2:
    w_var = st.selectbox("② 干擾變數（W）", options=mod_options, index=0, key="mod_w")

# dv options exclude IV & W
dv_options2 = [""] + [c for c in dim_cols if c not in {iv_w, w_var}]
with col3:
    dv_w = st.selectbox("③ 依變數（DV）", options=dv_options2, index=0, key="mod_dv")

chosen2 = [x for x in [iv_w, w_var, dv_w] if x]
if len(chosen2) != len(set(chosen2)):
    st.error("⚠️ IV / W / DV 不可重複，A、B、C、D… 每個只能出現在一個角色中。")
else:
    if iv_w and w_var and dv_w:
        st.success(f"干擾模型：{iv_w} → {dv_w}（W={w_var}）")

        st.markdown("### 研究用資料表（僅保留 IV / W / DV）")
        df_moderation = df_raw_plus_dimmeans[[iv_w, w_var, dv_w]].copy()
        st.dataframe(df_moderation, width="stretch")

        st.download_button(
            "下載 干擾分析研究用資料 CSV（IV + W + DV）",
            data=df_to_csv_bytes(df_moderation),
            file_name=f"moderation_dataset_{iv_w}_{w_var}_{dv_w}.csv",
            mime="text/csv",
        )

        run_mod = st.button("執行干擾分析", type="primary", key="run_moderation")

        if run_mod:
            try:
                mod_table, mod_meta = build_moderation_paper_table(
                    df_raw_plus_dimmeans, iv=iv_w, mod=w_var, dv=dv_w
                )

                # ✅ 你指定的標題
                st.markdown(
                    f"### 干擾變數（{w_var}）對 自變數（{iv_w}）與 依變數（{dv_w}）之干擾分析表"
                )

                st.dataframe(mod_table, width="stretch")
                st.caption("註：* P<0.05，** P<0.01，*** P<0.001；ΔR² 為 R² 變化量（R² change）。")

                tag2 = f"{iv_w}_x_{w_var}_to_{dv_w}".replace(" ", "")
                st.download_button(
                    "下載 干擾分析表 CSV",
                    data=df_to_csv_bytes(mod_table),
                    file_name=f"moderation_table_{tag2}.csv",
                    mime="text/csv",
                )

                st.markdown(f"**N={mod_meta['N']}**")

            except Exception as e:
                st.error("干擾分析失敗（safe）")
                safe_show_exception(e)

    else:
        st.info("請依序選擇 IV / W / DV（且三者不可重複）後，才會顯示干擾分析資料與結果。")



# ---- GPT report (optional) ----
st.divider()
st.subheader("📝 GPT 論文報告生成（文字）")

if not gpt_on:
    st.info("你目前未啟用 GPT 報告。若要生成論文文字，請在左側打開「啟用 GPT 報告」。")
    st.stop()

if not GPT_AVAILABLE:
    st.warning("找不到可用的 generate_gpt_report（請確認 gpt_report.py 中有定義 generate_gpt_report）。")
    st.stop()

key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
if not key:
    st.warning("尚未提供 OpenAI API Key。請在左側輸入，或設定環境變數 OPENAI_API_KEY。")
    st.stop()

gen = st.button("生成 GPT 報告（文字）", type="primary")

if gen:
    try:
        report = generate_gpt_report(result_df, model=model_name, api_key=key)

        paper_text = None
        if isinstance(report, dict):
            paper_text = report.get("paper_text") or report.get("text") or report.get("output")
        elif isinstance(report, str):
            paper_text = report

        if not paper_text:
            st.warning("GPT 回傳內容為空，請檢查 gpt_report.py 的回傳格式。")
        else:
            st.success("GPT 報告生成完成。")
            st.text_area("GPT 論文報告（可複製）", value=paper_text, height=420)

            st.download_button(
                "下載 GPT 報告 TXT",
                data=paper_text.encode("utf-8"),
                file_name="gpt_paper_report.txt",
                mime="text/plain",
            )

    except Exception as e:
        msg = repr(e)
        if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
            st.error("GPT report failed：你的 OpenAI API 帳號目前沒有可用額度（insufficient_quota）。")
            st.caption("解法：到 OpenAI 平台 Billing/Credits 加值後再試。")
        else:
            st.error("GPT report failed. See error details below (safe).")
            safe_show_exception(e)
