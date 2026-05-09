from __future__ import annotations

import base64
import mimetypes
import sys
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui_helpers import (  # noqa: E402
    SUPPORTED_PATCHES,
    build_champion_diagnostics,
    champion_icon_path,
    get_available_champions,
    run_optimizer_for_gui,
)
from data_loader import load_patch_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"


st.set_page_config(
    page_title="LoL Mid Pool Optimizer",
    layout="wide",
)

if "candidate_selection" in st.query_params:
    del st.query_params["candidate_selection"]
    st.rerun()


@st.cache_data(show_spinner=False)
def load_champion_options(patch: str) -> list[str]:
    return get_available_champions(patch, DATA_DIR)


@st.cache_data(show_spinner=False)
def load_champion_pickrates(patch: str) -> dict[str, float]:
    loaded = load_patch_data(patch, DATA_DIR)
    if loaded.summary_df.empty or "pickrate" not in loaded.summary_df.columns:
        return {}
    return {
        str(row.champion_name): float(row.pickrate)
        for row in loaded.summary_df.itertuples(index=False)
        if pd.notna(row.pickrate)
    }


@st.cache_data(show_spinner=True)
def run_cached_optimizer(
    patch: str,
    candidates: tuple[str, ...],
    pool_size: int,
    top_k: int,
):
    return run_optimizer_for_gui(
        patch=patch,
        data_dir=DATA_DIR,
        candidates=list(candidates),
        pool_size=pool_size,
        top_k=top_k,
    )


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def color_signed_delta(value: float) -> str:
    if pd.isna(value) or abs(value) < 1e-12:
        return "color: inherit;"
    if value < 0:
        return "color: #f87171;"
    return "color: #4ade80;"


def format_signed_percent(value: float) -> str:
    if pd.isna(value) or abs(value) < 0.0000005:
        return "0.00%"
    return f"{value:+.2%}"


def render_champion_card(champion: str, score: float | None = None, role: str | None = None) -> None:
    icon_path = champion_icon_path(champion, PROJECT_ROOT)
    with st.container(border=True):
        if icon_path:
            st.image(str(icon_path), use_container_width=True)
        st.markdown(f"**{champion}**")
        if role:
            st.caption(role)
        if score is not None:
            st.caption(format_percent(score))


def render_small_champion_icon(champion: str, size: int = 20) -> None:
    icon_path = champion_icon_path(champion, PROJECT_ROOT)
    if icon_path:
        st.image(str(icon_path), width=size)
        return
    initials = "".join(part[:1] for part in champion.split()[:2]).upper() or "?"
    st.markdown(
        f"""
        <div style="
            width:{size}px;height:{size}px;border-radius:5px;
            display:flex;align-items:center;justify-content:center;
            background:#1f2937;color:#d1d5db;
            font-size:0.62rem;font-weight:700;
            border:1px solid rgba(255,255,255,0.12);
        ">{escape(initials)}</div>
        """,
        unsafe_allow_html=True,
    )


def champion_icon_data_uri(champion: str) -> str | None:
    icon_path = champion_icon_path(champion, PROJECT_ROOT)
    if not icon_path:
        return None
    mime_type = mimetypes.guess_type(icon_path)[0] or "image/png"
    encoded = base64.b64encode(icon_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def champion_label_html(champion: str, icon_size: int = 24) -> str:
    icon_uri = champion_icon_data_uri(champion)
    if icon_uri:
        icon_html = (
            f'<img class="matchup-icon" src="{icon_uri}" '
            f'alt="{escape(champion)}" width="{icon_size}" height="{icon_size}">'
        )
    else:
        initials = "".join(part[:1] for part in champion.split()[:2]).upper() or "?"
        icon_html = (
            f'<span class="matchup-icon matchup-placeholder" '
            f'style="width:{icon_size}px;height:{icon_size}px;">{escape(initials)}</span>'
        )
    return (
        '<div class="matchup-label">'
        f"{icon_html}"
        f'<span class="matchup-name">{escape(champion)}</span>'
        "</div>"
    )


def picker_icon_html(champion: str, icon_size: int = 24) -> str:
    icon_uri = champion_icon_data_uri(champion)
    if icon_uri:
        return (
            f'<img class="picker-row-icon" src="{icon_uri}" '
            f'alt="{escape(champion)}" width="{icon_size}" height="{icon_size}">'
        )
    initials = "".join(part[:1] for part in champion.split()[:2]).upper() or "?"
    return (
        f'<span class="picker-row-icon picker-row-placeholder" '
        f'style="width:{icon_size}px;height:{icon_size}px;">{escape(initials)}</span>'
    )


def render_champion_picker(
    available_champions: list[str],
    pickrates: dict[str, float],
    patch: str,
) -> list[str]:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            --chip-height: 20px;
            --row-height: 24px;
            --picker-icon-size: 16px;
            --picker-button-size: 18px;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            min-height: var(--picker-button-size);
            height: var(--picker-button-size);
            padding: 0 0.22rem;
            font-size: 0.66rem;
            line-height: 1;
            border-radius: 3px;
        }
        section[data-testid="stSidebar"] div[data-testid="stImage"] img {
            border-radius: 3px;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.08rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 0.12rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            margin: 0;
        }
        section[data-testid="stSidebar"] div[data-testid="stElementContainer"] {
            margin: 0;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 3px;
            border-color: rgba(255,255,255,0.14);
            background: rgba(17,24,39,0.30);
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.02rem 0.08rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"] {
            min-height: 22px;
            padding: 1px 3px;
            margin: 1px 0;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 3px;
            background: rgba(31,41,55,0.62);
            display: flex;
            align-items: center;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="column"] {
            display: flex;
            align-items: center;
            min-height: 18px;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 18px;
            margin: 0;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] button {
            width: 18px;
            min-width: 18px;
            padding: 0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
            min-height: 1.55rem;
            height: 1.55rem;
            padding: 0.10rem 0.48rem;
            font-size: 0.74rem;
        }
        .champion-picker-note {
            color: #9ca3af;
            font-size: 0.70rem;
            margin: 0.02rem 0 0.08rem;
        }
        .picker-row-icon {
            width: var(--picker-icon-size);
            height: var(--picker-icon-size);
            border-radius: 3px;
            object-fit: cover;
            border: 1px solid rgba(255,255,255,0.14);
            background: #111827;
        }
        .picker-row-placeholder {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #d1d5db;
            font-size: 0.56rem;
            font-weight: 750;
        }
        .picker-row-name {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #e5e7eb;
            font-size: 0.69rem;
            font-weight: 650;
            line-height: 1;
        }
        .picker-row-stat {
            color: #9ca3af;
            font-size: 0.64rem;
            text-align: right;
            font-variant-numeric: tabular-nums;
            line-height: 1;
        }
        .picker-selected-name {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #e5e7eb;
            font-size: 0.66rem;
            font-weight: 600;
            line-height: 1;
        }
        .pool-picker-shell {
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 3px;
            background: rgba(39,39,42,0.78);
            padding: 4px;
            margin: 2px 0 4px;
        }
        .selected-chip-grid {
            max-height: 74px;
            overflow-y: auto;
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            align-items: flex-start;
        }
        .selected-chip {
            height: 21px;
            display: inline-grid;
            grid-template-columns: 16px auto 14px;
            align-items: center;
            gap: 4px;
            padding: 1px 4px;
            border: 1px solid rgba(255,255,255,0.24);
            border-radius: 2px;
            background: #3f3f46;
            color: #f8fafc;
            text-decoration: none !important;
            font-size: 0.70rem;
            font-weight: 650;
            line-height: 1;
        }
        .selected-chip:hover {
            background: #52525b;
            border-color: rgba(255,255,255,0.34);
            color: #ffffff;
        }
        .selected-chip span,
        .selected-chip:hover span,
        .available-row-reference,
        .available-row-reference:hover,
        .available-row-reference div {
            text-decoration: none !important;
        }
        .selected-chip .picker-row-icon {
            width: 16px;
            height: 16px;
            border-radius: 2px;
        }
        .selected-chip-remove {
            color: #e5e7eb;
            font-size: 0.72rem;
            line-height: 1;
            text-align: center;
        }
        .available-list-reference {
            max-height: 210px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 3px;
            background: rgba(31,31,31,0.76);
            padding: 4px 5px;
        }
        .available-row-reference {
            height: 24px;
            display: grid;
            grid-template-columns: 18px minmax(0, 1fr) 46px;
            align-items: center;
            gap: 7px;
            color: #f8fafc;
            text-decoration: none;
            border-radius: 2px;
            padding: 1px 2px;
        }
        .available-row-reference:hover {
            background: rgba(255,255,255,0.06);
        }
        .available-row-reference .picker-row-icon {
            width: 18px;
            height: 18px;
            border-radius: 2px;
        }
        .available-row-reference .picker-row-name {
            font-size: 0.76rem;
            color: #f8fafc;
        }
        .available-row-reference .picker-row-stat {
            font-size: 0.70rem;
            color: #9ca3af;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Candidate champions**")

    include_col, exclude_col = st.columns(2)
    with include_col:
        if st.button("Include all", use_container_width=True):
            st.session_state["selected_candidates"] = list(available_champions)
            st.rerun()
    with exclude_col:
        if st.button("Exclude all", use_container_width=True):
            st.session_state["selected_candidates"] = []
            st.rerun()

    selected = [
        champion
        for champion in st.session_state.get("selected_candidates", [])
        if champion in available_champions
    ]
    selected_set = set(selected)
    st.caption(f"Selected candidates: {len(selected)} / {len(available_champions)}")

    if not selected:
        st.markdown('<div class="champion-picker-note">No champions selected.</div>', unsafe_allow_html=True)
    else:
        selected_box = st.container(height=106, border=True)
        with selected_box:
            for index, champion in enumerate(selected):
                icon_col, name_col, remove_col = st.columns(
                    [0.10, 0.76, 0.14],
                    vertical_alignment="center",
                )
                with icon_col:
                    st.markdown(picker_icon_html(champion, icon_size=16), unsafe_allow_html=True)
                with name_col:
                    st.markdown(
                        f'<div class="picker-row-name">{escape(champion)}</div>',
                        unsafe_allow_html=True,
                    )
                with remove_col:
                    if st.button(
                        "×",
                        key=f"remove_candidate_{patch}_{index}_{champion}",
                        help=f"Remove {champion}",
                    ):
                        st.session_state["selected_candidates"] = [
                            item for item in selected if item != champion
                        ]
                        st.rerun()

    search = st.text_input(
        "Add champions",
        key=f"candidate_search_{patch}",
        placeholder="type to add...",
        label_visibility="collapsed",
    )
    search_text = search.strip().lower()
    matching_champions = [
        champion
        for champion in available_champions
        if champion not in selected_set and search_text in champion.lower()
    ]

    available_box = st.container(height=210, border=True)
    with available_box:
        if not matching_champions:
            st.caption("No available champions match your search.")
        for index, champion in enumerate(matching_champions):
            pickrate = pickrates.get(champion)
            stat = format_percent(pickrate) if pickrate is not None else ""
            icon_col, name_col, stat_col, add_col = st.columns(
                [0.10, 0.52, 0.24, 0.14],
                vertical_alignment="center",
            )
            with icon_col:
                st.markdown(picker_icon_html(champion, icon_size=16), unsafe_allow_html=True)
            with name_col:
                st.markdown(f'<div class="picker-row-name">{escape(champion)}</div>', unsafe_allow_html=True)
            with stat_col:
                st.markdown(f'<div class="picker-row-stat">{escape(stat)}</div>', unsafe_allow_html=True)
            with add_col:
                if st.button("+", key=f"add_candidate_{patch}_{index}_{champion}", help=f"Add {champion}"):
                    st.session_state["selected_candidates"] = selected + [champion]
                    st.rerun()

    return st.session_state["selected_candidates"]


def matchup_cell_background(value: float) -> str:
    if pd.isna(value):
        return "#374151"
    distance = min(abs(value - 0.50) / 0.12, 1.0)
    if value < 0.50:
        red = int(72 + (96 * distance))
        green = int(73 - (29 * distance))
        blue = int(70 - (34 * distance))
        return f"rgb({red}, {green}, {blue})"
    red = int(63 - (31 * distance))
    green = int(79 + (69 * distance))
    blue = int(68 - (21 * distance))
    return f"rgb({red}, {green}, {blue})"


def render_counterpick_heatmap_matrix(result) -> None:
    heatmap_df = result.heatmap_data.copy()
    if heatmap_df.empty:
        st.warning("No heatmap data is available for the selected pool.")
        return

    value_df = heatmap_df.set_index("enemy_champion")[list(result.best_pool)]
    st.caption("Rows = enemy champions, columns = champions in selected pool")

    column_count = len(value_df.columns)
    html = [
        "<style>",
        ".matchup-legend { display:flex; gap:8px; align-items:center; margin: 0.2rem 0 0.55rem; color:#9ca3af; font-size:0.78rem; }",
        ".legend-chip { width:34px; height:10px; border-radius:999px; display:inline-block; border:1px solid rgba(255,255,255,0.12); }",
        ".legend-weak { background:#a82c24; }",
        ".legend-even { background:#3f4f44; }",
        ".legend-strong { background:#20942f; }",
        ".counterpick-matrix-wrap {",
        "  overflow: hidden;",
        "  border: 1px solid rgba(255,255,255,0.10);",
        "  border-radius: 10px;",
        "  background: #111827;",
        "}",
        ".counterpick-horizontal-scroll {",
        "  overflow-x: auto;",
        "}",
        ".counterpick-header-grid, .counterpick-body-grid {",
        "  display: grid;",
        f"  grid-template-columns: minmax(168px, 1.2fr) repeat({column_count}, minmax(96px, 1fr));",
        "  gap: 5px;",
        "  align-items: center;",
        "  min-width: max-content;",
        "}",
        ".counterpick-header-grid {",
        "  background: #111827;",
        "  padding: 8px 8px 10px;",
        "  border-bottom: 1px solid rgba(255,255,255,0.10);",
        "  box-shadow: 0 4px 10px rgba(0,0,0,0.22);",
        "}",
        ".counterpick-body-scroll {",
        "  max-height: 640px;",
        "  overflow-y: auto;",
        "  overflow-x: hidden;",
        "  padding: 8px;",
        "}",
        ".counterpick-corner, .counterpick-header {",
        "  background: #111827;",
        "}",
        ".counterpick-row-label {",
        "  background: linear-gradient(90deg, #111827 88%, rgba(17,24,39,0.78));",
        "  border-radius: 8px;",
        "  padding: 4px 8px 4px 2px;",
        "}",
        ".counterpick-cell {",
        "  min-height: 34px;",
        "  border-radius: 8px;",
        "  display: flex;",
        "  align-items: center;",
        "  justify-content: center;",
        "  color: #f9fafb;",
        "  font-size: 0.83rem;",
        "  font-weight: 700;",
        "  border: 1px solid rgba(255,255,255,0.08);",
        "  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);",
        "}",
        ".counterpick-cell.best {",
        "  border: 2px solid #020617;",
        "  box-shadow: 0 0 0 1px rgba(255,255,255,0.35), inset 0 1px 0 rgba(255,255,255,0.16);",
        "}",
        ".counterpick-cell.missing { color: #9ca3af; font-weight: 600; }",
        ".matchup-label { display:flex; align-items:center; gap:7px; min-width:0; }",
        ".counterpick-header .matchup-label { justify-content:center; flex-direction:column; gap:4px; }",
        ".matchup-icon { border-radius:6px; object-fit:cover; flex:0 0 auto; border:1px solid rgba(255,255,255,0.18); background:#1f2937; }",
        ".matchup-placeholder { display:inline-flex; align-items:center; justify-content:center; color:#d1d5db; font-size:0.62rem; font-weight:750; }",
        ".matchup-name { color:#e5e7eb; font-size:0.78rem; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:130px; }",
        ".counterpick-header .matchup-name { max-width:92px; text-align:center; font-size:0.72rem; }",
        "</style>",
        '<div class="matchup-legend">',
        '<span>Weak</span><span class="legend-chip legend-weak"></span>',
        '<span>Even</span><span class="legend-chip legend-even"></span>',
        '<span>Strong</span><span class="legend-chip legend-strong"></span>',
        "</div>",
        '<div class="counterpick-matrix-wrap">',
        '<div class="counterpick-horizontal-scroll">',
        '<div class="counterpick-header-grid">',
        '<div class="counterpick-corner"></div>',
    ]
    html.extend(
        f'<div class="counterpick-header">{champion_label_html(str(champion), icon_size=26)}</div>'
        for champion in value_df.columns
    )
    html.extend(["</div>", '<div class="counterpick-body-scroll">', '<div class="counterpick-body-grid">'])

    for enemy_champion, row in value_df.iterrows():
        best_value = row.max(skipna=True)
        html.append(
            f'<div class="counterpick-row-label">{champion_label_html(str(enemy_champion), icon_size=24)}</div>'
        )
        for pool_champion, value in row.items():
            is_missing = pd.isna(value)
            is_best = pd.notna(value) and pd.notna(best_value) and value == best_value
            classes = "counterpick-cell"
            if is_best:
                classes += " best"
            if is_missing:
                classes += " missing"
            label = "n/a" if is_missing else f"{value:.1%}"
            tooltip = (
                f"Enemy: {enemy_champion} | Pool champion: {pool_champion} | "
                f"Winrate: {label}"
            )
            if is_best:
                tooltip += " | Best option in row"
            html.append(
                f'<div class="{classes}" style="background:{matchup_cell_background(value)};" '
                f'title="{escape(tooltip)}">'
                f"{escape(label)}</div>"
            )

    html.extend(["</div>", "</div>", "</div>", "</div>"])
    st.markdown("\n".join(html), unsafe_allow_html=True)


def render_exclusion_details(result) -> None:
    details = result.exclusion_details
    if not details["had_exclusions"]:
        return

    st.warning("Some matchup rows were excluded during scoring. Click below for details.")
    with st.expander("Show excluded matchup rows"):
        summary_cols = st.columns(5)
        summary_cols[0].metric("Total skipped rows", details["total_skipped"])
        summary_cols[1].metric("Self matchups", details["self_matchups"])
        summary_cols[2].metric("Missing matchup values", details["missing_matchups"])
        summary_cols[3].metric("Missing frequencies", details["missing_frequencies"])
        summary_cols[4].metric(
            "Removed frequency mass",
            format_percent(details["removed_frequency_mass"]),
        )
        st.caption(
            "Remaining scorable enemy frequency mass was renormalized afterward, matching the existing scoring logic."
        )

        skipped_rows = details["skipped_rows"].copy()
        if skipped_rows.empty:
            st.info("No skipped row details are available.")
            return
        skipped_rows["original_frequency"] = skipped_rows["original_frequency"].map(format_percent)
        st.dataframe(
            skipped_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "pool_champion": "Pool Champion",
                "enemy_champion": "Enemy Champion",
                "reason": "Reason",
                "original_frequency": "Original Frequency",
                "notes": "Notes",
            },
        )


st.title("LoL Midlane Champion Pool Optimizer")
st.caption("Local prototype GUI over the existing patch-based optimizer.")

with st.sidebar:
    st.header("Optimizer Inputs")
    patch = st.selectbox("Patch", SUPPORTED_PATCHES, index=len(SUPPORTED_PATCHES) - 1)

    try:
        available_champions = load_champion_options(patch)
        champion_pickrates = load_champion_pickrates(patch)
    except Exception as exc:
        st.error(f"Could not load patch data: {exc}")
        st.stop()

    previous_patch = st.session_state.get("selected_candidates_patch")
    if "selected_candidates" not in st.session_state:
        st.session_state["selected_candidates"] = list(available_champions)
    elif previous_patch != patch:
        st.session_state["selected_candidates"] = [
            champion
            for champion in st.session_state["selected_candidates"]
            if champion in available_champions
        ]
    st.session_state["selected_candidates_patch"] = patch

    pool_size = st.number_input("Pool size", min_value=1, max_value=10, value=3, step=1)
    candidates = render_champion_picker(
        available_champions=available_champions,
        pickrates=champion_pickrates,
        patch=patch,
    )

    show_advanced = st.checkbox("Show advanced diagnostics", value=False)
    run_button = st.button("Run optimizer", type="primary", use_container_width=True)

if not run_button:
    st.info("Choose a patch, candidate set, and pool size, then run the optimizer.")
    st.stop()

if not candidates:
    st.error("Select at least one candidate champion.")
    st.stop()
if pool_size > len(candidates):
    st.error(f"Pool size {pool_size} is larger than the {len(candidates)} selected candidates.")
    st.stop()

try:
    result = run_cached_optimizer(patch, tuple(sorted(candidates)), int(pool_size), 5)
except Exception as exc:
    st.error(f"Optimizer could not run: {exc}")
    st.stop()

top_cols = st.columns(5)
top_cols[0].metric("Best Pool Score", format_percent(result.best_pool_score))
top_cols[1].metric("Best Blind Pick", result.best_blind_pick)
top_cols[2].metric("Blind Score", format_percent(result.best_blind_score))
top_cols[3].metric("Patch", result.loaded.patch_label)
top_cols[4].metric("Candidates", len(result.candidates))

st.subheader("Best Pool")
blind_lookup = dict(zip(result.blind_scores["champion"], result.blind_scores["blind_score"]))
card_cols = st.columns(len(result.best_pool))
for col, champion in zip(card_cols, result.best_pool):
    with col:
        role = "Best blind pick" if champion == result.best_blind_pick else "Counterpick option"
        render_champion_card(champion, blind_lookup.get(champion), role)

render_exclusion_details(result)

st.subheader("Top 5 Pools")
top_pools_display = result.top_pools[
    ["rank", "pool_champions", "pool_score", "difference_from_best"]
].copy()
top_pools_display["difference_from_best"] = (
    top_pools_display["pool_score"] - result.best_pool_score
)
top_pools_styled = top_pools_display.style.format(
    {
        "pool_score": "{:.2%}",
        "difference_from_best": format_signed_percent,
    }
).map(color_signed_delta, subset=["difference_from_best"])
st.dataframe(
    top_pools_styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "rank": "Rank",
        "pool_champions": "Pool Champions",
        "pool_score": "Score",
        "difference_from_best": "Difference From Best",
    },
)
st.caption(
    "The best few pools often have very similar scores, so small score differences should be interpreted cautiously."
)

st.subheader("Counterpick Heatmap")
render_counterpick_heatmap_matrix(result)

st.subheader("Pool Responsibility")
if result.pool_responsibility.empty:
    st.warning("No responsibility data is available for this pool.")
else:
    fig_resp = px.bar(
        result.pool_responsibility,
        x="champion",
        y="weighted_share",
        text=result.pool_responsibility["weighted_share"].map(lambda value: f"{value:.1%}"),
        labels={"champion": "Pool Champion", "weighted_share": "Weighted Enemy Share Covered"},
    )
    fig_resp.update_traces(textposition="outside")
    fig_resp.update_layout(yaxis_tickformat=".0%", height=360)
    st.plotly_chart(fig_resp, use_container_width=True)

st.subheader("Champion Diagnostics")
diagnostic_scope = sorted(set(result.candidates).union(set(available_champions)))
selected_champion = st.selectbox("Champion", diagnostic_scope, index=diagnostic_scope.index(result.best_blind_pick))
diagnostics = build_champion_diagnostics(
    selected_champion,
    loaded=result.loaded,
    candidates=result.candidates,
)

summary = diagnostics["summary"]
diag_cols = st.columns(4)
diag_cols[0].metric("Total / Blind Score", format_percent(diagnostics["blind_score"]))
diag_cols[1].metric("Pickrate", format_percent(summary.get("pickrate")))
diag_cols[2].metric("Banrate", format_percent(summary.get("banrate")))
diag_cols[3].metric("Total Games", f"{int(summary['total_games']):,}" if "total_games" in summary and pd.notna(summary["total_games"]) else "n/a")

extra_cols = st.columns(3)
extra_cols[0].metric("Depth", f"{summary.get('depth'):.3f}" if summary.get("depth") is not None and pd.notna(summary.get("depth")) else "n/a")
extra_cols[1].metric("Worst10 Mean", format_percent(summary.get("worst10_mean")))
extra_cols[2].metric("Weighted CVaR 10", format_percent(summary.get("weighted_cvar_10")))

contrib = diagnostics["top_matchup_contributions"]
if contrib.empty:
    st.warning("This champion has no matchup contribution data for the selected patch.")
else:
    fig_contrib = px.bar(
        contrib.sort_values("contribution", ascending=True),
        x="contribution",
        y="enemy_champion",
        orientation="h",
        labels={"contribution": "f_j * W(i,j)", "enemy_champion": "Enemy Champion"},
    )
    fig_contrib.update_layout(height=420)
    st.plotly_chart(fig_contrib, use_container_width=True)

if show_advanced:
    with st.expander("Advanced profile analysis", expanded=True):
        profile = diagnostics["profile"]
        if profile.empty:
            st.info("Baseline profile data is not available for this champion.")
        else:
            profile_top = profile.head(20)
            profile_long = profile_top.melt(
                id_vars="enemy_champion",
                value_vars=["baseline_contribution", "champion_contribution"],
                var_name="series",
                value_name="contribution",
            )
            fig_profile = px.line(
                profile_long,
                x="enemy_champion",
                y="contribution",
                color="series",
                markers=True,
                labels={"enemy_champion": "Enemy Champion", "contribution": "Contribution"},
            )
            fig_profile.update_layout(height=420)
            st.plotly_chart(fig_profile, use_container_width=True)

            fig_delta = px.bar(
                profile_top,
                x="enemy_champion",
                y="delta",
                labels={"enemy_champion": "Enemy Champion", "delta": "Champion Minus Baseline"},
            )
            fig_delta.update_layout(height=320)
            st.plotly_chart(fig_delta, use_container_width=True)
