# core/io.py
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st


# -----------------------------
# Template File Loader (NEW)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # folder containing app.py
TEMPLATE_ROOT = PROJECT_ROOT / "input_templates"


@st.cache_data(show_spinner=False)
def load_template_bytes(template_relative_path: str) -> bytes:
    """
    Load an .xlsx template from input_templates/ and return raw bytes.
    Example: template_relative_path = "manual_journals_template.xlsx"
    """
    path = TEMPLATE_ROOT / template_relative_path
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_bytes()


# -----------------------------
# Existing helpers (keep these)
# -----------------------------
def make_template_bytes(sheet_name: str, columns: list[str]) -> bytes:
    """Fallback: Create a blank Excel template if you want to keep generator support."""
    df = pd.DataFrame(columns=columns)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buf.seek(0)
    return buf.getvalue()


def load_excel(uploaded_file):
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
        return None


def build_reports_excel(reports: Dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for sheet, df in reports.items():
            safe = (sheet or "Sheet")[:31]
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_excel(
                writer, sheet_name=safe, index=False
            )
    buf.seek(0)
    return buf.getvalue()