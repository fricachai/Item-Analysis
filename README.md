# 柴康偉 論文統計分析專業版

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 執行

```bash
streamlit run app.py
```

## 本版修正重點

- 修正區別效度表的子構面判斷邏輯。
- D1、D2、D3 不再被合併成 D。
- E1、E2、F1 等後續構面皆可依同一規則自動產生。
- A11、A12、A13 仍歸為 A1；A21、A22、A23 仍歸為 A2。
- app.py 與 analysis.py 的 `_subdim_code()` 已統一。
