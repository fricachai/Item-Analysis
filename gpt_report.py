import os
import pandas as pd
from openai import OpenAI


def _build_prompt(result_df: pd.DataFrame) -> str:
    # 只取必要欄位，避免 token 爆掉；欄位名稱對齊本系統輸出的 Item Analysis 表。
    keep = [
        "構面", "子構面", "題項", "平均數", "標準差", "CITC", "因素負荷量",
        "刪除後 Cronbach α", "該子構面整體 α", "警示標記",
        "決斷值(CR: Critical Ratio)", "CR_p值",
    ]
    cols = [c for c in keep if c in result_df.columns]
    view = result_df[cols].copy()

    table_text = view.to_csv(index=False, sep="\t")

    prompt = (
        "你是一位熟悉量表建構與信效度檢核的學術研究助理。\n"
        "請依據下列表格的 item analysis 結果，生成可放入論文的『結果敘述』，包含：\n"
        "1) 各子構面整體信度（Cronbach's α）摘要\n"
        "2) 題項鑑別度、CR_p值與 CITC 的判讀\n"
        "3) 因素負荷量的整體狀況與可能需修正的題項（根據警示標記）\n"
        "4) 一段方法限制說明（例如同源偏誤、樣本、PCA限制等）\n\n"
        "【Item analysis 表格（TSV）】\n"
        f"{table_text}\n"
    )
    return prompt


def generate_gpt_report(result_df: pd.DataFrame, model: str = "gpt-4o-mini", api_key: str | None = None) -> str:
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("找不到 OPENAI_API_KEY。請在系統左側輸入 API Key，或先設定環境變數 OPENAI_API_KEY。")

    client = OpenAI(api_key=key)
    prompt = _build_prompt(result_df)

    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    text = getattr(resp, "output_text", None)
    if text:
        return text

    try:
        chunks = []
        for item in resp.output:
            for c in item.content:
                if getattr(c, "type", "") == "output_text":
                    chunks.append(c.text)
        return "\n".join(chunks).strip()
    except Exception:
        return str(resp)
