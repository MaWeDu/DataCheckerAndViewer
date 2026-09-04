import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# STREAMLIT KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Quality Analyzer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Universeller Data Quality Analyzer")
st.caption(
    "CSV- und Excel-Dateien analysieren: Struktur, Datenqualität, "
    "Statistik, IQR-Ausreißer, Visualisierungen und optionales Profiling."
)


# ============================================================
# PROFILING OPTIONAL IMPORTIEREN
# ============================================================

try:
    from data_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    ProfileReport = None
    PROFILING_AVAILABLE = False


# ============================================================
# DATEI EINLESEN
# ============================================================

@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes, suffix):
    if suffix not in {".xlsx", ".xls"}:
        return []

    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    return excel_file.sheet_names


@st.cache_data(show_spinner=False)
def load_data_file(file_bytes, filename, sheet_name=None):
    """
    Lädt CSV-, XLSX- oder XLS-Dateien aus Bytes.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".csv":
        buffer = io.BytesIO(file_bytes)

        try:
            return pd.read_csv(buffer, encoding="utf-8")
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="cp1252")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name,
        )

    raise ValueError(
        "Nicht unterstütztes Dateiformat. "
        "Erlaubt sind CSV, XLSX und XLS."
    )


# ============================================================
# DATUMSFORMAT-ERKENNUNG
# ============================================================

def detect_date_pattern(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    patterns = {
        "YYYY-MM-DD": r"^\d{4}-\d{2}-\d{2}$",
        "YYYY/MM/DD": r"^\d{4}/\d{2}/\d{2}$",
        "YYYY.MM.DD": r"^\d{4}\.\d{2}\.\d{2}$",
        "DD-MM-YYYY or MM-DD-YYYY": r"^\d{2}-\d{2}-\d{4}$",
        "DD/MM/YYYY or MM/DD/YYYY": r"^\d{2}/\d{2}/\d{4}$",
        "Month DD, YYYY": r"^[A-Za-z]+ \d{1,2}, \d{4}$",
    }

    for name, pattern in patterns.items():
        if re.match(pattern, value):
            return name

    return None


# ============================================================
# STRUKTURPRÜFUNG UND HEADER-KORREKTUR
# ============================================================

def looks_like_bad_header(columns):
    columns = [str(col).strip() for col in columns]

    if not columns:
        return False

    suspicious = 0

    for col in columns:
        col_lower = col.lower()

        if (
            col_lower.startswith("unnamed:")
            or re.fullmatch(r"column[_ ]?\d+", col_lower)
            or re.fullmatch(r"field[_ ]?\d+", col_lower)
            or re.fullmatch(r"\d+", col_lower)
            or col_lower in {"", "nan", "none", "null"}
        ):
            suspicious += 1

    return (suspicious / len(columns)) >= 0.8


def row_looks_like_header(row):
    values = [str(v).strip() for v in row.tolist()]

    if not values:
        return False

    non_empty = [
        v for v in values
        if v.lower() not in {"", "nan", "none", "null"}
    ]

    if len(non_empty) < max(2, int(len(values) * 0.7)):
        return False

    numeric_count = 0
    long_count = 0

    for value in non_empty:
        try:
            float(value.replace(",", "."))
            numeric_count += 1
        except ValueError:
            pass

        if len(value) > 80:
            long_count += 1

    numeric_ratio = numeric_count / len(non_empty)
    unique_ratio = len(set(non_empty)) / len(non_empty)
    long_ratio = long_count / len(non_empty)

    return (
        numeric_ratio <= 0.35
        and unique_ratio >= 0.8
        and long_ratio <= 0.2
    )


def fix_misplaced_header(df):
    structure_report = []

    bad_header = looks_like_bad_header(df.columns)

    structure_report.append({
        "check": "suspicious_header",
        "detected": bad_header,
        "details": str(list(df.columns)),
    })

    if not bad_header or df.empty:
        return df, pd.DataFrame(structure_report)

    first_row_header = row_looks_like_header(df.iloc[0])

    structure_report.append({
        "check": "first_row_looks_like_header",
        "detected": first_row_header,
        "details": str(df.iloc[0].astype(str).str.strip().tolist()),
    })

    if not first_row_header:
        return df, pd.DataFrame(structure_report)

    new_columns = (
        df.iloc[0]
        .astype(str)
        .str.strip()
        .tolist()
    )

    valid_columns = all(
        col and col.lower() not in {"nan", "none", "null"}
        for col in new_columns
    )

    unique_columns = len(new_columns) == len(set(new_columns))

    if not (valid_columns and unique_columns):
        structure_report.append({
            "check": "header_correction",
            "detected": False,
            "details": (
                "Korrektur verworfen: "
                "leere oder doppelte Spaltennamen."
            ),
        })
        return df, pd.DataFrame(structure_report)

    df = df.iloc[1:].copy()
    df.columns = new_columns
    df.reset_index(drop=True, inplace=True)

    structure_report.append({
        "check": "header_correction",
        "detected": True,
        "details": f"Header korrigiert. Neue Spalten: {new_columns}",
    })

    return df, pd.DataFrame(structure_report)


# ============================================================
# DATENQUALITÄTSANALYSE
# ============================================================

def analyze_data_quality(df, sample_size=10):
    results = []

    missing_tokens = {
        "nan", "na", "n/a", "none", "null",
        "missing", "-", "--", "unknown"
    }

    for col in df.columns:
        s = df[col]
        s_str = s.astype("string")

        results.append({
            "column": col,
            "check": "dtype",
            "count": len(s),
            "examples": str(s.dtype),
        })

        results.append({
            "column": col,
            "check": "missing_nan",
            "count": int(s.isna().sum()),
            "examples": None,
        })

        non_null = s_str.dropna()

        if non_null.empty:
            continue

        mask = non_null.str.strip().eq("")
        if mask.any():
            results.append({
                "column": col,
                "check": "empty_string",
                "count": int(mask.sum()),
                "examples": non_null[mask].head(sample_size).tolist(),
            })

        mask = non_null.ne(non_null.str.strip())
        if mask.any():
            results.append({
                "column": col,
                "check": "leading_trailing_whitespace",
                "count": int(mask.sum()),
                "examples": (
                    non_null[mask]
                    .drop_duplicates()
                    .head(sample_size)
                    .tolist()
                ),
            })

        mask = non_null.str.contains(r"\s{2,}", regex=True)
        if mask.any():
            results.append({
                "column": col,
                "check": "multiple_spaces",
                "count": int(mask.sum()),
                "examples": (
                    non_null[mask]
                    .drop_duplicates()
                    .head(sample_size)
                    .tolist()
                ),
            })

        normalized = non_null.str.strip().str.lower()

        mask = normalized.isin(missing_tokens)
        if mask.any():
            results.append({
                "column": col,
                "check": "missing_value_as_text",
                "count": int(mask.sum()),
                "examples": (
                    non_null[mask]
                    .drop_duplicates()
                    .head(sample_size)
                    .tolist()
                ),
            })

        temp = pd.DataFrame({
            "original": non_null,
            "normalized": non_null.str.strip().str.lower(),
        })

        variants = (
            temp.groupby("normalized")["original"]
            .nunique()
        )

        problematic = variants[variants > 1].index

        if len(problematic) > 0:
            examples = (
                temp[temp["normalized"].isin(problematic)]
                ["original"]
                .drop_duplicates()
                .head(sample_size)
                .tolist()
            )

            results.append({
                "column": col,
                "check": "case_variations",
                "count": len(problematic),
                "examples": examples,
            })

        cleaned = non_null.str.strip()

        numeric = pd.to_numeric(
            cleaned.str.replace(",", ".", regex=False),
            errors="coerce",
        )

        numeric_ratio = numeric.notna().mean()

        if numeric_ratio >= 0.5:
            invalid_numeric = (
                numeric.isna()
                & ~cleaned.str.lower().isin(missing_tokens)
                & cleaned.ne("")
            )

            if invalid_numeric.any():
                results.append({
                    "column": col,
                    "check": "invalid_numeric_value",
                    "count": int(invalid_numeric.sum()),
                    "examples": (
                        cleaned[invalid_numeric]
                        .drop_duplicates()
                        .head(sample_size)
                        .tolist()
                    ),
                })

        try:
            date_parsed = pd.to_datetime(
                cleaned,
                errors="coerce",
                format="mixed",
            )
        except (TypeError, ValueError):
            date_parsed = pd.to_datetime(
                cleaned,
                errors="coerce",
            )

        date_ratio = date_parsed.notna().mean()

        if date_ratio >= 0.5:
            invalid_date = (
                date_parsed.isna()
                & ~cleaned.str.lower().isin(missing_tokens)
                & cleaned.ne("")
            )

            if invalid_date.any():
                results.append({
                    "column": col,
                    "check": "invalid_date",
                    "count": int(invalid_date.sum()),
                    "examples": (
                        cleaned[invalid_date]
                        .drop_duplicates()
                        .head(sample_size)
                        .tolist()
                    ),
                })

            date_patterns = cleaned.apply(detect_date_pattern)

            unique_patterns = (
                date_patterns[date_patterns.notna()]
                .unique()
            )

            if len(unique_patterns) > 1:
                results.append({
                    "column": col,
                    "check": "mixed_date_formats",
                    "count": len(unique_patterns),
                    "examples": list(unique_patterns),
                })

        value_counts = cleaned.value_counts()

        rare_values = value_counts[
            value_counts <= max(2, len(cleaned) * 0.001)
        ]

        if len(rare_values) > 0:
            results.append({
                "column": col,
                "check": "rare_values",
                "count": len(rare_values),
                "examples": (
                    rare_values
                    .head(sample_size)
                    .index
                    .tolist()
                ),
            })

    return pd.DataFrame(results)


# ============================================================
# STATISTIK UND AUSREISSER
# ============================================================

def looks_like_id_column(series, column_name):
    name = str(column_name).strip().lower()

    name_is_id = (
        name == "id"
        or name.endswith("_id")
        or name.startswith("id_")
        or name.endswith(" id")
        or name.startswith("id ")
        or "uuid" in name
        or "identifier" in name
        or name == "key"
        or name.endswith("_key")
    )

    s = series.dropna()

    if s.empty:
        return name_is_id

    s_str = s.astype(str).str.strip()

    uuid_pattern = re.compile(
        r"^[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[1-5][0-9a-fA-F]{3}-"
        r"[89abAB][0-9a-fA-F]{3}-"
        r"[0-9a-fA-F]{12}$"
    )

    uuid_ratio = s_str.str.match(uuid_pattern).mean()

    return name_is_id or uuid_ratio >= 0.8


def get_numeric_series(df, column, numeric_threshold=0.5):
    if column not in df.columns:
        return None

    if looks_like_id_column(df[column], column):
        return None

    original = df[column].dropna()

    if original.empty:
        return None

    cleaned = (
        original
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )

    numeric = pd.to_numeric(cleaned, errors="coerce")

    if numeric.notna().mean() < numeric_threshold:
        return None

    numeric = numeric.dropna()

    if numeric.empty:
        return None

    return numeric


def get_numeric_columns(df, numeric_threshold=0.5):
    columns = []

    for col in df.columns:
        numeric = get_numeric_series(
            df,
            col,
            numeric_threshold=numeric_threshold,
        )

        if numeric is not None:
            columns.append(col)

    return columns


def numeric_statistics(df, numeric_threshold=0.5):
    results = []

    for col in df.columns:
        if looks_like_id_column(df[col], col):
            continue

        original_non_null = df[col].dropna()

        if original_non_null.empty:
            continue

        cleaned = (
            original_non_null
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        numeric = pd.to_numeric(cleaned, errors="coerce")
        numeric_ratio = numeric.notna().mean()

        if numeric_ratio < numeric_threshold:
            continue

        values = numeric.dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        modes = values.mode()

        if modes.empty:
            mode_value = np.nan
        elif len(modes) == 1:
            mode_value = modes.iloc[0]
        else:
            mode_value = ", ".join(
                str(v) for v in modes.head(5).tolist()
            )

        results.append({
            "column": col,
            "count": int(values.count()),
            "missing": int(len(df) - values.count()),
            "numeric_ratio": round(float(numeric_ratio), 4),
            "mean": values.mean(),
            "median": values.median(),
            "mode": mode_value,
            "min": values.min(),
            "max": values.max(),
            "range": values.max() - values.min(),
            "variance": values.var(),
            "std": values.std(),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
        })

    return pd.DataFrame(results)


def detect_iqr_outliers(df, numeric_threshold=0.5, sample_size=10):
    results = []

    for col in df.columns:
        if looks_like_id_column(df[col], col):
            continue

        original_non_null = df[col].dropna()

        if original_non_null.empty:
            continue

        cleaned = (
            original_non_null
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        numeric = pd.to_numeric(cleaned, errors="coerce")
        numeric_ratio = numeric.notna().mean()

        if numeric_ratio < numeric_threshold:
            continue

        values = numeric.dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        outliers = values[outlier_mask]

        if outliers.empty:
            continue

        results.append({
            "column": col,
            "check": "possible_outlier_iqr",
            "count": int(outliers.count()),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "examples": outliers.head(sample_size).tolist(),
            "row_indices": outliers.index[:sample_size].tolist(),
        })

    return pd.DataFrame(results)


# ============================================================
# VISUALISIERUNG
# ============================================================

def make_histogram(values, column, bins=30):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(values, bins=bins)
    ax.set_title(f"Verteilung – {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Häufigkeit")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def make_boxplot(values, column, lower_bound=None, upper_bound=None, title=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.boxplot(
        values,
        vert=False,
        tick_labels=[column],
    )

    if lower_bound is not None:
        ax.axvline(
            lower_bound,
            linestyle="--",
            label=f"Untere Grenze: {lower_bound:.2f}",
        )

    if upper_bound is not None:
        ax.axvline(
            upper_bound,
            linestyle="--",
            label=f"Obere Grenze: {upper_bound:.2f}",
        )

    if lower_bound is not None or upper_bound is not None:
        ax.legend()

    ax.set_title(title or f"Boxplot und IQR-Ausreißeranalyse – {column}")
    ax.set_xlabel("Wert")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    return fig


# ============================================================
# DOWNLOAD-HILFSFUNKTIONEN
# ============================================================

def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def create_profile_html(df, filename, minimal=False):
    if not PROFILING_AVAILABLE:
        return None

    base_name = Path(filename).stem

    profile = ProfileReport(
        df,
        title=f"Data Profiling Report - {base_name}",
        explorative=True,
        minimal=minimal,
    )

    return profile.to_html()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Einstellungen")

    numeric_threshold = st.slider(
        "Schwellwert für numerische Spalten",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "Anteil erfolgreich numerisch interpretierbarer Werte, "
            "ab dem eine Spalte als numerisch behandelt wird."
        ),
    )

    sample_size = st.slider(
        "Maximale Beispielwerte",
        min_value=3,
        max_value=25,
        value=10,
        step=1,
    )

    bins = st.slider(
        "Histogramm-Bins",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
    )

    st.divider()

    if PROFILING_AVAILABLE:
        st.success("fg-data-profiling verfügbar")
    else:
        st.info(
            "fg-data-profiling ist nicht installiert. "
            "Die übrige App funktioniert trotzdem."
        )


# ============================================================
# DATEI-UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "CSV- oder Excel-Datei auswählen",
    type=["csv", "xlsx", "xls"],
    help="Unterstützt werden .csv, .xlsx und .xls",
)

if uploaded_file is None:
    st.info("Bitte eine CSV- oder Excel-Datei hochladen.")
    st.stop()

file_bytes = uploaded_file.getvalue()
suffix = Path(uploaded_file.name).suffix.lower()

sheet_name = None

if suffix in {".xlsx", ".xls"}:
    try:
        sheet_names = get_excel_sheet_names(
            file_bytes,
            suffix,
        )
    except Exception as exc:
        st.error(f"Excel-Datei konnte nicht gelesen werden: {exc}")
        st.stop()

    if not sheet_names:
        st.error("In der Excel-Datei wurden keine Tabellenblätter gefunden.")
        st.stop()

    sheet_name = st.selectbox(
        "Tabellenblatt auswählen",
        options=sheet_names,
    )

try:
    with st.spinner("Datei wird eingelesen ..."):
        raw_df = load_data_file(
            file_bytes,
            uploaded_file.name,
            sheet_name=sheet_name,
        )
except Exception as exc:
    st.error(f"Fehler beim Einlesen der Datei: {exc}")
    st.stop()

df, structure_report = fix_misplaced_header(raw_df.copy())


# ============================================================
# ANALYSE
# ============================================================

with st.spinner("Daten werden analysiert ..."):
    quality_report = analyze_data_quality(
        df,
        sample_size=sample_size,
    )

    statistics_report = numeric_statistics(
        df,
        numeric_threshold=numeric_threshold,
    )

    outlier_report = detect_iqr_outliers(
        df,
        numeric_threshold=numeric_threshold,
        sample_size=sample_size,
    )

numeric_columns = get_numeric_columns(
    df,
    numeric_threshold=numeric_threshold,
)


# ============================================================
# ÜBERSICHT
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Zeilen", f"{df.shape[0]:,}".replace(",", "."))
col2.metric("Spalten", df.shape[1])
col3.metric("Numerische Spalten", len(numeric_columns))
col4.metric(
    "Spalten mit IQR-Ausreißern",
    0 if outlier_report.empty else len(outlier_report),
)

if len(raw_df) != len(df) or list(raw_df.columns) != list(df.columns):
    st.warning(
        "Eine wahrscheinlich verschobene Header-Zeile wurde automatisch korrigiert."
    )

st.caption(
    f"Datei: {uploaded_file.name}"
    + (
        f" · Tabellenblatt: {sheet_name}"
        if sheet_name is not None
        else ""
    )
)


# ============================================================
# TABS
# ============================================================

tab_data, tab_structure, tab_quality, tab_stats, tab_outliers, tab_charts, tab_profile = st.tabs(
    [
        "Daten",
        "Struktur",
        "Datenqualität",
        "Statistik",
        "Ausreißer",
        "Grafiken",
        "Profiling",
    ]
)


# ------------------------------------------------------------
# DATEN
# ------------------------------------------------------------

with tab_data:
    st.subheader("Datenvorschau")

    preview_rows = st.number_input(
        "Anzahl Vorschauzeilen",
        min_value=5,
        max_value=min(max(len(df), 5), 500),
        value=min(50, max(len(df), 5)),
        step=5,
    )

    st.dataframe(
        df.head(int(preview_rows)),
        use_container_width=True,
    )

    st.download_button(
        "Analysierte Daten als CSV herunterladen",
        data=dataframe_to_csv_bytes(df),
        file_name=f"{Path(uploaded_file.name).stem}_analysiert.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# STRUKTUR
# ------------------------------------------------------------

with tab_structure:
    st.subheader("Strukturprüfung")

    st.dataframe(
        structure_report,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Spalten**")
    st.dataframe(
        pd.DataFrame({
            "Nr.": range(1, len(df.columns) + 1),
            "Spalte": df.columns,
            "Datentyp": [str(df[col].dtype) for col in df.columns],
        }),
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------
# DATENQUALITÄT
# ------------------------------------------------------------

with tab_quality:
    st.subheader("Datenqualitätsanalyse")

    filter_col1, filter_col2 = st.columns(2)

    quality_columns = ["Alle"] + sorted(
        quality_report["column"].dropna().astype(str).unique().tolist()
    )
    quality_checks = ["Alle"] + sorted(
        quality_report["check"].dropna().astype(str).unique().tolist()
    )

    selected_column = filter_col1.selectbox(
        "Spalte filtern",
        quality_columns,
    )

    selected_check = filter_col2.selectbox(
        "Prüfung filtern",
        quality_checks,
    )

    filtered_quality = quality_report.copy()

    if selected_column != "Alle":
        filtered_quality = filtered_quality[
            filtered_quality["column"].astype(str) == selected_column
        ]

    if selected_check != "Alle":
        filtered_quality = filtered_quality[
            filtered_quality["check"].astype(str) == selected_check
        ]

    st.dataframe(
        filtered_quality,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Datenqualitätsreport als CSV herunterladen",
        data=dataframe_to_csv_bytes(quality_report),
        file_name=f"{Path(uploaded_file.name).stem}_data_quality.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# STATISTIK
# ------------------------------------------------------------

with tab_stats:
    st.subheader("Statistische Kennzahlen")

    if statistics_report.empty:
        st.info("Keine geeigneten numerischen Spalten gefunden.")
    else:
        st.dataframe(
            statistics_report,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Statistik als CSV herunterladen",
            data=dataframe_to_csv_bytes(statistics_report),
            file_name=f"{Path(uploaded_file.name).stem}_statistik.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# AUSREISSER
# ------------------------------------------------------------

with tab_outliers:
    st.subheader("IQR-Ausreißeranalyse")
    st.caption(
        "Die IQR-Methode markiert statistisch auffällige Werte. "
        "Ein Ausreißer ist nicht automatisch ein Datenfehler."
    )

    if outlier_report.empty:
        st.success("Keine IQR-Ausreißer gefunden.")
    else:
        st.dataframe(
            outlier_report,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Ausreißerreport als CSV herunterladen",
            data=dataframe_to_csv_bytes(outlier_report),
            file_name=f"{Path(uploaded_file.name).stem}_ausreisser.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# GRAFIKEN
# ------------------------------------------------------------

with tab_charts:
    st.subheader("Grafische Datenanalyse")

    if not numeric_columns:
        st.info("Keine geeigneten numerischen Spalten gefunden.")
    else:
        selected_chart_columns = st.multiselect(
            "Numerische Spalten auswählen",
            options=numeric_columns,
            default=numeric_columns[: min(5, len(numeric_columns))],
        )

        chart_type = st.radio(
            "Darstellung",
            options=[
                "Histogramm",
                "Boxplot mit IQR-Grenzen",
                "Beides",
            ],
            horizontal=True,
        )

        for col in selected_chart_columns:
            values = get_numeric_series(
                df,
                col,
                numeric_threshold=numeric_threshold,
            )

            if values is None or values.empty:
                continue

            if chart_type in {"Histogramm", "Beides"}:
                fig = make_histogram(
                    values,
                    col,
                    bins=bins,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            if chart_type in {"Boxplot mit IQR-Grenzen", "Beides"}:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1

                lower_bound = (
                    q1 - 1.5 * iqr
                    if iqr > 0
                    else None
                )
                upper_bound = (
                    q3 + 1.5 * iqr
                    if iqr > 0
                    else None
                )

                fig = make_boxplot(
                    values,
                    col,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ------------------------------------------------------------
# PROFILING
# ------------------------------------------------------------

with tab_profile:
    st.subheader("Automatisches Datenprofiling")

    if not PROFILING_AVAILABLE:
        st.warning(
            "Für diesen Bereich muss `fg-data-profiling` installiert sein. "
            "Die übrigen Analysen der App stehen bereits zur Verfügung."
        )
        st.code(
            "pip install streamlit pandas numpy matplotlib "
            "openpyxl xlrd \"fg-data-profiling[notebook]\"",
            language="bash",
        )
    else:
        minimal_default = len(df) > 100000

        minimal_profile = st.checkbox(
            "Minimalen Profiling-Report verwenden",
            value=minimal_default,
            help=(
                "Für große Dateien ist der minimale Report "
                "deutlich schneller."
            ),
        )

        if st.button(
            "Profiling-Report erstellen",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Profiling-Report wird erstellt. "
                    "Dies kann je nach Dateigröße dauern ..."
                ):
                    profile_html = create_profile_html(
                        df,
                        uploaded_file.name,
                        minimal=minimal_profile,
                    )

                st.session_state["profile_html"] = profile_html
                st.success("Profiling-Report wurde erstellt.")

            except Exception as exc:
                st.error(
                    f"Profiling-Report konnte nicht erstellt werden: {exc}"
                )

        profile_html = st.session_state.get("profile_html")

        if profile_html:
            st.download_button(
                "Profiling-Report als HTML herunterladen",
                data=profile_html.encode("utf-8"),
                file_name=(
                    f"{Path(uploaded_file.name).stem}"
                    "_profiling_report.html"
                ),
                mime="text/html",
            )

            show_inline = st.checkbox(
                "Profiling-Report direkt in der App anzeigen",
                value=False,
            )

            if show_inline:
                import streamlit.components.v1 as components

                components.html(
                    profile_html,
                    height=900,
                    scrolling=True,
                )
