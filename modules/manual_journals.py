from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from core.types import Module, TestSpec, DataFrame


class ManualJournalsModule(Module):
    name = "Manual Journals Testing"
    input_template_path = "manual_journals_template.xlsx"

    BASE_REQUIRED = [
        "Entity",
        "Company Name",
        "Journal ID",
        "Transaction Date",
        "Posted Date",
        "User",
        "Transaction Amount",
        "Debit",
        "Credit",
    ]

    # Used in outputs (you referenced self.PERIOD_LABEL but it wasn’t defined)
    PERIOD_LABEL = "Count of Journals"

    # These are computed ONCE in run_tests and reused in every test
    _dfx: Optional[pd.DataFrame] = None
    _grouped: Optional[pd.DataFrame] = None

    # Charts created inside test functions are stored here
    _charts: Dict[str, "px.Figure"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._charts = {}

    # -----------------------------
    # 1) Prepare dataframe once (called ONLY from run_tests)
    # -----------------------------
    def _prepare_df(
        self,
        df: Optional[pd.DataFrame],
        required_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Return (prepared_df, error_df)."""
        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame([{"Info": "No data uploaded."}])

        dfx = df.copy()
        dfx.columns = [str(c).strip() for c in dfx.columns]

        # Drop Excel artifact columns
        unnamed = [c for c in dfx.columns if str(c).strip().lower().startswith("unnamed")]
        if unnamed:
            dfx = dfx.drop(columns=unnamed, errors="ignore")

        needed = required_cols or self.BASE_REQUIRED
        missing = [c for c in needed if c not in dfx.columns]
        if missing:
            return (
                pd.DataFrame(),
                pd.DataFrame([{
                    "Error": "Missing required columns",
                    "Missing Columns": ", ".join(missing),
                    "Available Columns": ", ".join(map(str, dfx.columns)),
                }]),
            )

        # Parse dates
        dfx["Transaction Date"] = pd.to_datetime(dfx["Transaction Date"], errors="coerce", dayfirst=True)
        dfx["Posted Date"] = pd.to_datetime(dfx["Posted Date"], errors="coerce", dayfirst=True)

        # Clean user
        dfx["User"] = (
            dfx["User"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )

        # Numeric coercion
        for col in ["Transaction Amount", "Debit", "Credit"]:
            dfx[col] = pd.to_numeric(dfx[col], errors="coerce")

        debit_all_empty = dfx["Debit"].isna().all() or (dfx["Debit"].fillna(0) == 0).all()
        credit_all_empty = dfx["Credit"].isna().all() or (dfx["Credit"].fillna(0) == 0).all()

        amt = dfx["Transaction Amount"]

        def fill_from_amt(amount_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
            a = amount_series.fillna(0)
            debit = np.where(a > 0, a, 0.0)
            credit = np.where(a < 0, -a, 0.0)
            return debit, credit

        # If both debit and credit are basically missing, derive them from Transaction Amount
        if debit_all_empty and credit_all_empty:
            d, c = fill_from_amt(amt)
            dfx["Debit"] = d
            dfx["Credit"] = c
        else:
            # Fill debit/credit when both are missing but amount exists
            mask = (
                (dfx["Debit"].isna() | (dfx["Debit"].fillna(0) == 0)) &
                (dfx["Credit"].isna() | (dfx["Credit"].fillna(0) == 0)) &
                (amt.notna())
            )
            if mask.any():
                d, c = fill_from_amt(amt.loc[mask])
                dfx.loc[mask, "Debit"] = d
                dfx.loc[mask, "Credit"] = c

        return dfx, None

    # -----------------------------
    # 2) Build grouped_journal_id once (called ONLY from run_tests)
    # -----------------------------
    def _build_grouped_journal_id(self, dfx: pd.DataFrame) -> pd.DataFrame:
        grouped = (
            dfx.groupby("Journal ID", dropna=False)
            .agg(
                **{
                    "Journal Value": ("Debit", "sum"),  # matches your original
                    "Transaction Date": ("Transaction Date", "first"),
                    "Posted Date": ("Posted Date", "first"),
                    "User": ("User", "first"),
                }
            )
            .reset_index()
        )

        # Day/Week/Month features
        grouped["day"] = grouped["Posted Date"].dt.day
        grouped["day_name"] = grouped["Posted Date"].dt.day_name()
        grouped["week_number"] = grouped["Posted Date"].dt.isocalendar().week.astype("Int64")
        grouped["month"] = grouped["Posted Date"].dt.month
        grouped["month_name"] = grouped["Posted Date"].dt.month_name()
        grouped["days_between_post_transaction"] = (grouped["Posted Date"] - grouped["Transaction Date"]).dt.days

        # Formatted columns (keep raw datetime cols too)
        grouped["Posted Date_formatted"] = grouped["Posted Date"].dt.strftime("%d/%m/%Y")
        grouped["Transaction Date_formatted"] = grouped["Transaction Date"].dt.strftime("%d/%m/%Y")

        return grouped

    # -----------------------------
    # Optional helper: attach chart to returned DF (non-breaking)
    # -----------------------------
    def _attach_chart(self, out_df: pd.DataFrame, key: str, fig: "px.Figure") -> pd.DataFrame:
        self._charts[key] = fig
        try:
            charts = out_df.attrs.get("charts", {})
            charts[key] = fig
            out_df.attrs["charts"] = charts
        except Exception:
            pass
        return out_df

    @property
    def charts(self) -> Dict[str, "px.Figure"]:
        """Charts produced during the last run_tests() call."""
        return self._charts

    # =============================
    #          TESTS
    # =============================
    def _test_overview_stats(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        dfx = self._dfx
        grouped = self._grouped
        if dfx is None or grouped is None or dfx.empty or grouped.empty:
            return pd.DataFrame([{"Info": "No prepared data available."}])

        return pd.DataFrame([{
            "# of Journals": len(grouped),
            "# of blank Journals": dfx["Journal ID"].isna().sum(),
            "Debit Value": round(dfx["Debit"].sum(), 2),
            "Credit Value": round(dfx["Credit"].sum(), 2),
            "Earliest Transaction Date": str(grouped["Transaction Date"].min()),
            "Latest Transaction Date": str(grouped["Transaction Date"].max()),
        }])
#----------------------------------------------------
    def _test_journals_by_month(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        grouped = self._grouped
        if grouped is None or grouped.empty:
            return pd.DataFrame([{"Info": "No prepared/grouped data available."}])

        date_col = "Posting Date"

        if date_col not in grouped.columns:
            return pd.DataFrame([{"Info": f"Missing required date column: {date_col}"}])

        # Ensure datetime
        g = grouped.copy()
        g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

        # Month-Year bucket (month start timestamp)
        g["month_start"] = g[date_col].dt.to_period("M").dt.to_timestamp()
        g["month_year"] = g["month_start"].dt.strftime("%b %Y")  # e.g. "Jan 2026"

        out = (
            g.groupby(["month_start", "month_year"], dropna=False)
            .agg(
                **{
                    self.PERIOD_LABEL: ("Journal ID", "size"),
                    "Sum of Journal Value": ("Journal Value", "sum"),
                }
            )
            .reset_index()
            .sort_values("month_start")
            .rename(columns={"month_year": "Manual journals by month"})
        )

        out = out[["Manual journals by month", self.PERIOD_LABEL, "Sum of Journal Value"]]
        out.index = range(1, len(out) + 1)

        # Plotly chart
        fig = px.line(
            out.reset_index(drop=True),
            x="Manual journals by month",
            y="Sum of Journal Value",
            title="Manual Journals – Sum of Journal Value by Month-Year",
            markers=True,
        )
        fig.update_layout(xaxis_tickangle=-45)

        out = self._attach_chart(out, "Journals Posted by Month-Year", fig)
        return out
#---------------------------------------------------
    def _test_journals_by_user(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        grouped = self._grouped
        if grouped is None or grouped.empty:
            return pd.DataFrame([{"Info": "No prepared/grouped data available."}])

        out = (
            grouped.groupby("User", dropna=False)
            .agg(
                **{
                    self.PERIOD_LABEL: ("Journal ID", "size"),
                    "Sum of Journal Value": ("Journal Value", "sum"),
                }
            )
            .reset_index()
            .rename(columns={"User": "Manual journals by user"})
            .sort_values("Sum of Journal Value", ascending=False)
            .reset_index(drop=True)
        )

        # ✅ Chart inside the same function (top 15 by value)
        top = out.head(15).copy()
        fig = px.bar(
            top,
            x="Manual journals by user",
            y="Sum of Journal Value",
            title="Manual Journals – Top 15 Users by Journal Value",
        )
        fig.update_layout(xaxis_tickangle=-45)

        out = self._attach_chart(out, "Manual Journals by User", fig)
        return out

    def _test_posting_lag_bins(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        grouped = self._grouped
        if grouped is None or grouped.empty:
            return pd.DataFrame([{"Info": "No prepared/grouped data available."}])

        bins = [0, 7, 14, 21, 28, 35, float("inf")]
        labels = ["0 - 7 days", "7 - 14 days", "14 - 21 days", "21 - 28 days", "28 - 35 days", "> 35 days"]

        tmp = grouped.copy()
        tmp["days_difference_group"] = pd.cut(
            tmp["days_between_post_transaction"],
            bins=bins,
            labels=labels,
            right=False,
        )

        out = (
            tmp.groupby("days_difference_group", dropna=False, observed=True)
            .agg(
                **{
                    self.PERIOD_LABEL: ("Journal ID", "size"),
                    "Sum of Journal Value": ("Journal Value", "sum"),
                }
            )
            .reset_index()
            .rename(columns={"days_difference_group": "Posting Lag (Bins)"})
        )
        out.index = range(1, len(out) + 1)

        # ✅ Chart inside the same function
        fig = px.bar(
            out.reset_index(drop=True),
            x="Posting Lag (Bins)",
            y=self.PERIOD_LABEL,
            title="Posting Lag – Count of Journals by Bin",
        )
        out = self._attach_chart(out, "Posting Lag (Bins)", fig)
        return out

    def _test_negative_days(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        grouped = self._grouped
        if grouped is None or grouped.empty:
            return pd.DataFrame([{"Info": "No prepared/grouped data available."}])

        out = grouped[grouped["days_between_post_transaction"] < 0].copy()
        if out.empty:
            return pd.DataFrame([{"Info": "No rows found where Transaction Date > Posted Date (negative lag)."}])
        return out.reset_index(drop=True)

    def _test_weekend(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        grouped = self._grouped
        if grouped is None or grouped.empty:
            return pd.DataFrame([{"Info": "No prepared/grouped data available."}])
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        counts = (
            grouped["day_name"]
            .value_counts(dropna=False)          # include NaN if any
            .reindex(day_order, fill_value=0)    # ensure all 7 days appear
            .rename_axis("Day")
            .reset_index(name="Count")
        )
        # chart (counts by day)
        fig = px.bar(counts, x="Day", y="Count", title="Journals – Count by Day")
        counts = self._attach_chart(counts, "Journals (Count by Day)", fig)
        return counts
    # -----------------------------
    # Tests list (Receivables style)
    # -----------------------------
    tests: List[TestSpec] = [
        TestSpec(id="overview_stats",     label="Overview Stats",             description="Summary statistics",            func=_test_overview_stats),
        TestSpec(id="by_month",           label="Journals Posted by Month",    description="Count/value by month",          func=_test_journals_by_month),
        TestSpec(id="by_user",            label="Manual Journals by User",     description="Count/value by user",           func=_test_journals_by_user),
        TestSpec(id="posting_lag_bins",   label="Posting Lag (Bins)",          description="Posting lag distribution",      func=_test_posting_lag_bins),
        TestSpec(id="negative_days",      label="Negative Posting Lag",        description="Transaction date > posted date", func=_test_negative_days),
        TestSpec(id="weekend",            label="Weekend Journals (Sat_Sun)",  description="Weekend postings",              func=_test_weekend),
    ]

    # -----------------------------
    # Runner (EXACT same pattern as receivables)
    # -----------------------------
    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        """
        Prepare once, group once, run only selected tests (like receivables.py).
        """
        self._charts = {}  # reset charts each run

        results: Dict[str, DataFrame] = {}

        # ✅ Prepare ONCE at the start and cache for tests
        dfx, err = self._prepare_df(df)
        if err is not None:
            results["Input Error"] = err
            self._dfx = None
            self._grouped = None
            return results

        self._dfx = dfx
        self._grouped = self._build_grouped_journal_id(dfx)

        # ✅ Receivables-style test execution
        for spec in self.tests:
            if spec.id in selected_test_ids:
                try:
                    results[spec.label] = spec.func(self, self._dfx)  # type: ignore[misc]
                except Exception as e:
                    results[f"{spec.label} (Error)"] = pd.DataFrame([{
                        "Test": spec.label,
                        "Error": str(e),
                    }])

        return results
