#
# app.py
from __future__ import annotations

from datetime import datetime
import streamlit as st
import pandas as pd

from config import PAGES, APP_TITLE, APP_STEPS_CAPTION

from core.io import load_template_bytes, make_template_bytes, load_excel, build_reports_excel

from core.state1 import state_key
from modules import REGISTRY


def get_required_columns(module) -> list[str]:
    """Return the list of required column names for this module (for mapping UI)."""
    return list(
        getattr(module, "BASE_REQUIRED", None)
        or getattr(module, "template_columns", [])
        or []
    )


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename uploaded columns to required names. mapping: required_col -> uploaded_col name."""
    if df is None or not mapping:
        return df
    reverse = {uploaded: required for required, uploaded in mapping.items() if uploaded}
    return df.rename(columns=reverse)


# -----------------------------
# --------- Page Setup --------
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# --------- App Body ----------
# -----------------------------

st.sidebar.title("Audit Test Apps")

# Get available modules from registry but keep sidebar order from config
available_module_names = [p for p in PAGES if p in REGISTRY]
if not available_module_names:
    st.error("No modules registered. Please check modules/__init__.py")
    st.stop()

current_page = st.sidebar.radio("Select a Module", available_module_names, index=0)
module = REGISTRY[current_page]

st.title(APP_TITLE)
st.caption(APP_STEPS_CAPTION)

# -----------------------------
# ------- UI Components -------
# -----------------------------

def render_module_ui(module_name: str):
    st.header(module_name)

    # ---------- 1) Download Template ----------


    st.subheader("Download the Input Format Template")
    st.caption("Use this template to copy & paste data in the expected format.")

    mod = REGISTRY[module_name]

    template_path = getattr(mod, "input_template_path", None)

    # Prefer stored template file
    if template_path:
        try:
            tpl_bytes = load_template_bytes(template_path)
            download_name = template_path.split("/")[-1]
        except FileNotFoundError as e:
            st.warning(str(e))
            # Optional fallback to generated template if file missing:
            tpl_bytes = make_template_bytes(module_name, mod.template_columns)
            download_name = f"{module_name.replace(' ', '_')}_Template.xlsx"
    else:
        # Optional fallback to generated template if not configured
        tpl_bytes = make_template_bytes(module_name, mod.template_columns)
        download_name = f"{module_name.replace(' ', '_')}_Template.xlsx"

    st.download_button(
        label=f"⬇️ Download {module_name} Template (.xlsx)",
        data=tpl_bytes,
        file_name=download_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )


    st.divider()

    # ---------- 2) Upload File ----------
    st.subheader("Upload Completed File")
    uploaded = st.file_uploader(
        "Choose an Excel file (.xlsx)",
        type=["xlsx"],
        key=state_key(module_name, "uploader"),
    )
    df = load_excel(uploaded) if uploaded else None
    if df is not None:
        st.success(f"Loaded {len(df):,} rows")
        st.dataframe(df.head(50), use_container_width=True, height=260)

    # ---------- 2b) Column mapping (map uploaded columns to required names) ----------
    required_columns = get_required_columns(mod)
    column_mapping: dict[str, str] = {}
    if df is not None and required_columns:
        uploaded_cols = [str(c).strip() for c in df.columns]
        mapping_key = state_key(module_name, "column_mapping")
        existing = st.session_state.get(mapping_key, {})
        # Default: same name if present, else first uploaded column
        for req in required_columns:
            if req in uploaded_cols:
                column_mapping[req] = existing.get(req, req)
            else:
                column_mapping[req] = existing.get(req, uploaded_cols[0] if uploaded_cols else "")
            if column_mapping[req] not in uploaded_cols and uploaded_cols:
                column_mapping[req] = uploaded_cols[0]

        with st.container(border=True):
            st.subheader("Map columns to required names")
            st.caption("If your file uses different column names, map each required column to the column in your file. Then run tests below.")
            ncols = min(3, len(required_columns)) or 1
            cols = st.columns(ncols)
            for i, req in enumerate(required_columns):
                with cols[i % ncols]:
                    default_idx = 0
                    if column_mapping.get(req) in uploaded_cols:
                        default_idx = uploaded_cols.index(column_mapping[req])
                    chosen = st.selectbox(
                        f"**{req}** ←",
                        options=uploaded_cols,
                        index=default_idx,
                        key=state_key(module_name, f"map_{req}"),
                        label_visibility="visible",
                    )
                    column_mapping[req] = chosen
            st.session_state[mapping_key] = column_mapping

    st.divider()

    # ---------- 3) Test Checkboxes ----------
    st.subheader("Select Tests to Perform")
    selected_ids: list[str] = []
    test_params: dict = {}
    tests = REGISTRY[module_name].tests

    with st.container(border=True):
        colA, colB = st.columns([1, 3])
        with colA:
            select_all = st.checkbox("Select all", key=state_key(module_name, "select_all"))

        for i, spec in enumerate(tests):
            default_val = select_all
            # Late Manual Journals (by time): show checkbox and input on same row inside section
            if spec.id == "late_journals_by_time" and module_name == "Manual Journals Testing":
                c1, c2 = st.columns([1, 2])
                with c1:
                    checked = st.checkbox(
                        spec.label,
                        value=default_val,
                        key=state_key(module_name, f"test_{spec.id}"),
                        help=spec.description,
                    )
                with c2:
                    if checked:
                        late_cutoff = st.text_input(
                            "Cutoff time (journals posted *after* this time)",
                            placeholder="e.g. 17:00 or 5:00 PM",
                            key=state_key(module_name, "late_journals_cutoff"),
                            help="24h: 17:00, 17:30:00 • 12h: 5:00 PM, 5:30:00 PM",
                            label_visibility="collapsed",
                        )
                        test_params["late_journals_after_time"] = late_cutoff or ""
                        st.caption("⚠️ Use a valid format: **17:00** or **5:00 PM**")
                    else:
                        late_cutoff = ""
            else:
                checked = st.checkbox(
                    spec.label,
                    value=default_val,
                    key=state_key(module_name, f"test_{spec.id}"),
                    help=spec.description,
                )
            if checked:
                selected_ids.append(spec.id)

    if not selected_ids:
        st.info("Pick at least one test to enable **Run Tests**.")

    # ---------- 4) Run ----------
    run_clicked = st.button(
        "▶️ Run Tests",
        type="primary",
        use_container_width=True,
        disabled=(len(selected_ids) == 0),
        key=state_key(module_name, "run"),
    )
    if run_clicked:
        module = REGISTRY[module_name]
        mapping = st.session_state.get(state_key(module_name, "column_mapping"), {})
        df_to_run = apply_column_mapping(df, mapping) if df is not None else None
        if module_name == "Manual Journals Testing":
            reports = module.run_tests(df_to_run, selected_ids, test_params=test_params)
        else:
            reports = module.run_tests(df_to_run, selected_ids)
        st.session_state[state_key(module_name, "reports")] = reports
        st.success("Tests performed. See results below.")

    st.divider()

    # ---------- 5) Display Results ----------
    st.subheader("Performed Tests")
    reports = st.session_state.get(state_key(module_name, "reports"), {})
    if not reports:
        st.info("No test results yet. Select tests and click **Run Tests** above.")
    else:
        total_rows = sum(len(v) for v in reports.values())
        st.caption(
            f"Generated **{len(reports)} sheet(s)** • Total rows across tests: **{total_rows:,}**"
        )

        for name, res_df in reports.items():
            with st.expander(f"📄 {name} ({len(res_df):,} rows)", expanded=False):
                st.dataframe(res_df, use_container_width=True, height=260)
                # Render any Plotly charts attached to this result (e.g. Manual Journals)
                result_charts = getattr(res_df, "attrs", {}).get("charts", {})
                for chart_name, fig in result_charts.items():
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---------- 6) Download Excel ----------
        st.subheader("Download Results Workbook")
        default_name = f"{module_name.replace(' ', '_')}_Results.xlsx"
        out_name = st.text_input(
            "Enter a file name",
            value=default_name,
            key=state_key(module_name, "outname"),
        )
        if out_name and not out_name.lower().endswith(".xlsx"):
            out_name += ".xlsx"

        xlsx_bytes = build_reports_excel(reports)
        st.download_button(
            "⬇️ Download Excel workbook",
            data=xlsx_bytes,
            file_name=out_name or default_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

# Render the chosen module's UI
render_module_ui(module.name)
