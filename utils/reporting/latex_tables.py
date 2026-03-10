"""
LaTeX table generation utilities.

Provides functions to create publication-ready LaTeX tables from pandas DataFrames,
with support for best-value highlighting and consistent formatting.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def highlight_best(
    series: pd.Series,
    direction: Literal["min", "max"] = "min",
    format_str: str = "\\textbf{{{:.4f}}}",
) -> pd.Series:
    """
    Highlight the best value in a series with LaTeX bold formatting.

    Args:
        series: Pandas Series with numeric values
        direction: "min" to highlight minimum, "max" to highlight maximum
        format_str: Format string for the highlighted value (must contain {:.4f} or similar)

    Returns:
        Series with values as strings, best value wrapped in \\textbf{}
    """
    if direction == "min":
        best_val = series.min()
    else:
        best_val = series.max()

    def format_value(x):
        if pd.isna(x):
            return "--"
        if x == best_val:
            return format_str.format(x)
        # Extract format from format_str (e.g., ".4f" from "\\textbf{{{:.4f}}}")
        import re

        match = re.search(r"\{:([^}]+)\}", format_str)
        if match:
            plain_format = "{:" + match.group(1) + "}"
            return plain_format.format(x)
        return f"{x:.4f}"

    return series.apply(format_value)  # type: ignore[return-value]


def highlight_multiple_columns(
    df: pd.DataFrame,
    columns: list[str],
    direction: Literal["min", "max"] = "min",
    format_str: str = "\\textbf{{{:.4f}}}",
) -> pd.DataFrame:
    """
    Highlight best values in multiple columns.

    Args:
        df: DataFrame to modify
        columns: List of column names to highlight
        direction: "min" or "max"
        format_str: Format string for highlighted values

    Returns:
        DataFrame with highlighted columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = highlight_best(df[col], direction=direction, format_str=format_str)
    return df


def create_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str | None = None,
    highlight_columns: list[str] | None = None,
    highlight_direction: Literal["min", "max"] = "min",
    float_format: str = "%.4f",
    escape: bool = False,
    position: str = "htbp",
) -> str:
    """
    Create a LaTeX table string from a DataFrame.

    Uses hlines instead of booktabs for consistent style with existing tables.

    Args:
        df: DataFrame to convert
        caption: Table caption
        label: LaTeX label (e.g., "tab:phase1_results")
        column_format: LaTeX column format (e.g., "l|cccc"). Auto-generated if None.
        highlight_columns: List of columns to highlight best values
        highlight_direction: "min" or "max" for highlighting
        float_format: Printf-style format for floats
        escape: Whether to escape LaTeX special characters
        position: Table position specifier

    Returns:
        LaTeX table string
    """
    df = df.copy()

    # Apply highlighting if requested
    if highlight_columns:
        # Determine format string from float_format
        import re

        match = re.search(r"\.(\d+)f", float_format)
        decimals = int(match.group(1)) if match else 4
        format_str = "\\textbf{{{:." + str(decimals) + "f}}}"

        df = highlight_multiple_columns(
            df, highlight_columns, direction=highlight_direction, format_str=format_str
        )

    # Generate column format if not provided
    if column_format is None:
        column_format = "l" + "c" * (len(df.columns) - 1)

    # Generate LaTeX using pandas
    latex_str = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        float_format=float_format,
        column_format=column_format,
        escape=escape,
        position=position,
    )

    # Replace booktabs commands with hlines (matching existing style)
    latex_str = latex_str.replace(r"\toprule", r"\hline")
    latex_str = latex_str.replace(r"\midrule", r"\hline")
    latex_str = latex_str.replace(r"\bottomrule", r"\hline")

    return latex_str


def save_table(
    latex_str: str,
    output_path: Path | str,
    warning_url: str | None = None,
) -> None:
    """
    Save LaTeX table to file with optional warning comment.

    Args:
        latex_str: LaTeX table string
        output_path: Path to save the .tex file
        warning_url: URL to include in "DO NOT EDIT" warning comment
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add warning comment if URL provided
    if warning_url:
        warning = f"% DO NOT EDIT THIS TABLE - change it in code, see {warning_url}\n"
        latex_str = warning + latex_str

    with open(output_path, "w") as f:
        f.write(latex_str)

    logger.info(f"Saved table to {output_path}")


def create_summary_row(
    df: pd.DataFrame,
    label: str = "Mean",
    numeric_columns: list[str] | None = None,
) -> pd.Series:
    """
    Create a summary row (e.g., mean) for numeric columns.

    Args:
        df: DataFrame to summarize
        label: Label for the first column
        numeric_columns: Columns to compute mean for (default: auto-detect)

    Returns:
        Series that can be appended to the DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

    summary = {}
    for col in df.columns:
        if col in numeric_columns:
            summary[col] = df[col].mean()
        else:
            summary[col] = label if col == df.columns[0] else ""

    return pd.Series(summary)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}\\%"


def format_scientific(value: float, decimals: int = 2) -> str:
    """Format a number in scientific notation for LaTeX."""
    if value == 0:
        return "0"
    import math

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.{decimals}f} \\times 10^{{{exponent}}}"


def save_dataframe_csv(
    df: pd.DataFrame,
    output_path: Path | str,
    index: bool = False,
) -> None:
    """
    Save DataFrame to CSV with logging.

    Args:
        df: DataFrame to save
        output_path: Path to save the CSV file
        index: Whether to include the index column (default: False)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    logger.info(f"Saved CSV to {output_path}")


def create_comparison_table(
    df: pd.DataFrame,
    metric_columns: list[str],
    id_columns: list[str],
    highlight_columns: list[str] | None = None,
    rank_by: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Create formatted comparison table with optional ranking.

    Args:
        df: Source DataFrame
        metric_columns: List of metric column names to include
        id_columns: List of identifier columns (e.g., Model, Regularization)
        highlight_columns: Columns to mark for highlighting (metadata only)
        rank_by: Column name to rank by (adds Rank column if provided)
        ascending: Sort order for ranking (True = lowest is best)

    Returns:
        Formatted DataFrame ready for display or LaTeX conversion
    """
    # Select relevant columns
    all_columns = id_columns + metric_columns
    available_columns = [c for c in all_columns if c in df.columns]
    result = df[available_columns].copy()

    # Add ranking if requested
    if rank_by and rank_by in result.columns:
        result = result.sort_values(rank_by, ascending=ascending)
        result.insert(0, "Rank", range(1, len(result) + 1))

    # Store highlight info as metadata (for use by LaTeX generator)
    if highlight_columns:
        result.attrs["highlight_columns"] = highlight_columns

    return result
