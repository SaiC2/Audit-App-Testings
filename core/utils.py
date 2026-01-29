# core/utils.py
from datetime import datetime
import pandas as pd
from typing import Optional


def placeholder_report_df(test_name: str, src_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    meta = {
        "Test Name": test_name,
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Input Rows": 0 if src_df is None else len(src_df),
    }
    if isinstance(src_df, pd.DataFrame) and not src_df.empty:
        preview = src_df.head(3).copy()
        preview.columns = [f"src::{c}" for c in preview.columns]
        meta_df = pd.DataFrame([meta])
        return pd.concat([meta_df, preview], axis=1)
    return pd.DataFrame([meta])