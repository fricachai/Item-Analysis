# analysis.py
import re
import numpy as np
import pandas as pd
from scipy import stats

# ✅ 支援欄名：
# - A11
# - A01.題目
# - A01 題目
# - A01題目（代碼後直接接中文字，也要能抓到）
CODE_RE_PLAIN = re.compile(r"^[A-Za-z]\d{2,3}$")
CODE_RE_FROM_TEXT = re.compile(r"^([A-Za-z]\d{2,3})(?:[\.、\s]|(?=[^\d]))")  # ★放寬：.、空白、或後面接非數字(中文字)


def normalize_item_columns(df: pd.DataFrame):
    """
    將欄名正規化成純代碼（A01/A11/A101...），並回傳 mapping：原欄名 -> 代碼
    若同代碼重複，會加 _2 _3...
    """
    mapping = {}
    used = set()
    new_cols = []

    for col in df.columns:
        s = str(col).strip()
        code = None

        if CODE_RE_PLAIN.match(s):
            code = s
        else:
            m = CODE_RE_FROM_TEXT.match(s)
            if m:
                code = m.group(1)

        if code is None:
            new_cols.append(s)
            continue

        base = code
        k = 2
        while code in used:
            code = f"{base}_{k}"
            k += 1

        used.add(code)
        mapping[s] = code
        new_cols.append(code)

    df_norm = df.copy()
    df_norm.columns = new_cols
    return df_norm, mapping


def _find_item_cols(df: pd.DataFrame):
    # 允許 A01 / A11 / A101 / A11_2
    return [c for c in df.columns if re.match(r"^[A-Za-z]\d{2,3}(_\d+)?$", str(c).strip())]


def _parse_dim_and_subdim(code: str):
    """
    子構面只要「左邊兩碼」：A0 / A1 / B3 ...
    - A01 -> A0
    - A11 -> A1
    - A101 -> A1（仍取第一個數字）
    """
    m = re.match(r"^([A-Za-z])(\d{2,3})", str(code).strip())
    if not m:
        return None, None
    dim = m.group(1).upper()
    first_digit = m.group(2)[0]  # 只取左邊第一位數字
    subdim = f"{dim}{first_digit}"
    return dim, subdim


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    x = df_items.to_numpy(dtype=float)
    x = x[~np.isnan(x).all(axis=1)]  # 移除全空列
    k = x.shape[1]
    if k < 2:
        return np.nan
    item_var = np.nanvar(x, axis=0, ddof=1)
    total = np.nansum(x, axis=1)
    total_var = np.nanvar(total, ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - np.nansum(item_var) / total_var)


def citc(df_items: pd.DataFrame, col: str) -> float:
    x = df_items.astype(float)
    item = x[col]
    total_ex = x.drop(columns=[col]).sum(axis=1, skipna=True)
    if item.nunique(dropna=True) <= 1 or total_ex.nunique(dropna=True) <= 1:
        return np.nan
    return float(item.corr(total_ex))


def pca_single_loading_abs(df_items: pd.DataFrame, col: str) -> float:
    """
    ✅ 不用 sklearn
    單因子 loading：以「第一主成分分數」與「各題標準化分數」的相關（取絕對值）
    → 結果一定非負（符合你「因素負荷量不要負的」）
    """
    x = df_items.apply(pd.to_numeric, errors="coerce").astype(float)

    # 至少要有足夠資料
    x2 = x.dropna()
    if x2.shape[0] < 3 or x2.shape[1] < 2:
        return np.nan

    # 標準化
    std = x2.std(ddof=0)
    std[std == 0] = np.nan
    z = (x2 - x2.mean()) / std
    z = z.dropna(axis=1, how="any")  # 去掉常數欄造成的 nan
    if col not in z.columns or z.shape[1] < 2:
        return np.nan

    # 第一主成分方向（用 SVD 取得第一主軸）
    # z = U S Vt, 第一主成分分數 ~ U[:,0]*S[0] = z @ V[:,0]
    try:
        _, _, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan

    v1 = vt[0, :]  # 第一主軸
    pc1 = z.to_numpy() @ v1  # 第一主成分分數

    # loading = corr(z[col], pc1)
    item_z = z[col].to_numpy()
    if np.nanstd(item_z) == 0 or np.nanstd(pc1) == 0:
        return np.nan

    r = np.corrcoef(item_z, pc1)[0, 1]
    if np.isnan(r):
        return np.nan
    return float(abs(r))


def discrimination_t(df_items: pd.DataFrame, col: str, q_low=0.27, q_high=0.73):
    """
    以「子構面總分」做高低分組，再對單一題做 Welch t-test
    """
    x = df_items.apply(pd.to_numeric, errors="coerce").astype(float)
    total = x.sum(axis=1, skipna=True)

    low_cut = total.quantile(q_low)
    high_cut = total.quantile(q_high)

    low = x.loc[total <= low_cut, col].dropna()
    high = x.loc[total >= high_cut, col].dropna()

    if len(low) < 2 or len(high) < 2:
        return np.nan, np.nan

    t, p = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")
    return float(t), float(p)


def fmt(x, n: int) -> str:
    """
    ✅ 固定小數位顯示；就算是整數也補 0
    ✅ NaN 顯示空字串（避免 Streamlit 顯示 None）
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return ""


def run_item_analysis(df: pd.DataFrame) -> pd.DataFrame:
    item_cols = _find_item_cols(df)
    if not item_cols:
        raise ValueError("找不到符合「字母+數字」命名的題項欄位（如 A11 / A01 / A01題目）。請先用 normalize_item_columns 正規化欄名。")

    df_items_all = df[item_cols].copy()

    # 依 dim/subdim 分群（只算一次，不要每題重算）
    groups = {}
    for c in item_cols:
        dim, subdim = _parse_dim_and_subdim(c)
        if dim is None:
            continue
        groups.setdefault((dim, subdim), []).append(c)

    rows = []
    for (dim, subdim), cols in groups.items():
        df_sub = df_items_all[cols].apply(pd.to_numeric, errors="coerce").astype(float)

        alpha_total = cronbach_alpha(df_sub)

        # 先算這個子構面每一題 loading（非負）
        loadings = {c: pca_single_loading_abs(df_sub, c) for c in cols}

        for c in cols:
            item = pd.to_numeric(df_sub[c], errors="coerce")

            mean = item.mean(skipna=True)
            sd = item.std(skipna=True, ddof=1)

            citc_v = citc(df_sub, c)
            loading = loadings.get(c, np.nan)

            alpha_if = cronbach_alpha(df_sub.drop(columns=[c])) if df_sub.shape[1] >= 2 else np.nan
            t, p = discrimination_t(df_sub, c)

            warn = []
            if np.isfinite(citc_v) and citc_v < 0.30:
                warn.append("CITC<.30")
            if np.isfinite(loading) and loading < 0.50:
                warn.append("loading<.50")
            if np.isfinite(alpha_total) and np.isfinite(alpha_if) and alpha_if > alpha_total:
                warn.append("刪題α↑")

            rows.append({
                "構面": dim,
                "子構面": subdim,  # ✅ A0/A1/...
                "題項": c,

                # ✅ 固定小數位（你指定的）
                "平均數": fmt(mean, 3),
                "標準差": fmt(sd, 4),
                "CITC": fmt(citc_v, 4),
                "因素負荷量": fmt(loading, 4),
                "刪除後 Cronbach α": fmt(alpha_if, 4),
                "該子構面整體 α": fmt(alpha_total, 4),

                "警示標記": "；".join(warn) if warn else "—",

                # ✅ 欄名改成你要的 + 固定小數位
                "決斷值(CR: Critical Ratio)": fmt(t, 4),
                "CR_p值": fmt(p, 4),
            })

    out = pd.DataFrame(rows)

    # ✅ 就算 rows 為空，也要保證欄位存在，避免 KeyError
    expected_cols = [
        "構面", "子構面", "題項",
        "平均數", "標準差", "CITC", "因素負荷量",
        "刪除後 Cronbach α", "該子構面整體 α",
        "警示標記", "決斷值(CR: Critical Ratio)", "CR_p值"
    ]
    for col in expected_cols:
        if col not in out.columns:
            out[col] = ""

    out = out[expected_cols].sort_values(by=["構面", "子構面", "題項"], kind="mergesort").reset_index(drop=True)
    return out
