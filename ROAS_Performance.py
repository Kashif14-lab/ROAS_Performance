# ROAS Analyzer — Multi-user (per-user ID via URL), persistent across hard refresh
# Run: streamlit run app.py

import os
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -----------------------------
# Page config + light styling
# -----------------------------
st.set_page_config(layout="wide")
st.markdown("""
<style>
.stTabs [data-testid="stTab"] button { font-size: 1.1rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilities: per-user "sid" via URL (using st.query_params only)
# -----------------------------
try:
    from streamlit_browser_storage import BrowserStorage
    _HAS_BS = True
except Exception:
    _HAS_BS = False

def get_or_set_sid():
    if _HAS_BS:
        try:
            bs = BrowserStorage(key="roas_analyzer_user_id")
            sid = bs.get("sid")
            if not sid:
                sid = str(uuid.uuid4())
                bs.set("sid", sid)
            return sid
        except Exception:
            pass  # fall through to query_params if plugin misbehaves

    # Fallback: URL query param (no extra deps)
    if "sid" in st.query_params and st.query_params["sid"]:
        return st.query_params["sid"]
    sid = str(uuid.uuid4())
    st.query_params["sid"] = sid
    return sid

SID = get_or_set_sid()

# -----------------------------
# Per-sid cache path helpers
# -----------------------------
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

def _sid_cache_path(sid: str) -> str:
    return os.path.join(DATA_DIR, f"{sid}_last_upload.csv")

# -----------------------------
# Robust break-even crossing
# -----------------------------
def predict_cross_day(days, values, target=1.0):
    """
    Predict break-even day robustly:
      1) Interpolate missing values
      2) Enforce non-decreasing series
      3) Linear interpolation for the crossing point
    Returns float day or None if not reached within given days.
    """
    if not days or values is None:
        return None
    x = np.array(days, dtype=float)
    y = pd.Series(values, dtype="float64")
    if y.isna().all():
        return None
    y = y.interpolate(limit_direction="both")
    y = np.maximum.accumulate(y.values)
    if y[0] >= target:
        return 0.0
    if np.nanmax(y) < target:
        return None
    idx = int(np.argmax(y >= target))
    if idx == 0:
        return 0.0
    x1, x2 = x[idx-1], x[idx]
    y1, y2 = y[idx-1], y[idx]
    if pd.isna(y1) or pd.isna(y2):
        return None
    if y2 == y1:
        return x2
    return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

# -----------------------------
# App string parser
# -----------------------------
def parse_app_fields(app_str):
    """
    Expected patterns like: PS_IOS_Game_Name or PS_Android_Game_Name
    Returns Series: app_platform ('iOS'/'Android'/raw) and app_game (string after second underscore)
    """
    s = str(app_str) if pd.notna(app_str) else ""
    parts = s.split("_")
    platform, game = "Unknown", s
    if len(parts) >= 3:
        plat_raw = parts[1].lower()
        if "ios" in plat_raw:
            platform = "iOS"
        elif "and" in plat_raw or "android" in plat_raw:
            platform = "Android"
        else:
            platform = parts[1]
        game = "_".join(parts[2:])
    return pd.Series({"app_platform": platform, "app_game": game})

# -----------------------------
# Data loading / preprocess
# -----------------------------
def _parse_roas_series(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('%','').replace(',','').strip()
    return pd.to_numeric(s, errors='coerce')/100.0

@st.cache_data
def load_data_from_path_or_file(path_or_file):
    if isinstance(path_or_file, str):
        df = pd.read_csv(path_or_file, low_memory=False)
    else:
        df = pd.read_csv(path_or_file, low_memory=False)

    # Clean columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    required = {"app", "channel", "campaign_network"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Parse platform + game
    parsed = df["app"].apply(parse_app_fields)
    df["app_platform"] = parsed["app_platform"]
    df["app_game"] = parsed["app_game"]

    # ROAS columns
    roas_columns = [c for c in df.columns if c.lower().startswith("roas_d")]
    if not roas_columns:
        raise ValueError("No ROAS columns found (expected roas_d0, roas_d3, roas_d7, ...).")
    for c in roas_columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(_parse_roas_series)

    df["max_roas"] = df[roas_columns].max(axis=1)

    # Days (ints) for UI
    roas_days = []
    for c in roas_columns:
        cl = c.lower()
        if cl == "roas_d0": continue
        try: roas_days.append(int(cl.replace("roas_d","")))
        except: pass
    roas_days = sorted(set(roas_days))
    return df, roas_columns, roas_days

# -----------------------------
# Session state init
# -----------------------------
if "show_analysis_page" not in st.session_state:
    st.session_state.show_analysis_page = False
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.roas_columns = None
    st.session_state.roas_days = None
    st.session_state.uploaded_filename = None

if st.session_state.df is not None:
    st.session_state.show_analysis_page = True

# Auto-load this SID's cache (if exists)
if st.session_state.df is None and os.path.exists(_sid_cache_path(SID)):
    try:
        df, roas_cols, roas_days = load_data_from_path_or_file(_sid_cache_path(SID))
        st.session_state.df = df
        st.session_state.roas_columns = roas_cols
        st.session_state.roas_days = roas_days
        st.session_state.uploaded_filename = "last_upload.csv"
        st.session_state.show_analysis_page = True
        st.info("Loaded your file from session cache.")
    except Exception as e:
        st.warning(f"Cached file could not be loaded: {e}")

# -----------------------------
# Intro / Upload Page
# -----------------------------
if not st.session_state.show_analysis_page:
    st.title("Campaign Performance Analyzer (ROAS)")
    st.write("Upload an Adjust CSV to analyze ROAS simply and persist.")

    st.markdown("---")
    report_link = "https://suite.adjust.com/datascape/report?utc_offset=%2B00%3A00&reattributed=all&attribution_source=first&attribution_type=all&ad_spend_mode=network&date_period=-92d%3A-1d&cohort_maturity=mature&sandbox=false&channel_id__in=%22partner_257%22%2C%22partner_7%22%2C%22partner_34%22%2C%22partner_182%22%2C%22partner_100%22%2C%22partner_369%22%2C%22partner_56%22%2C%22partner_490%22%2C%22partner_2337%2C1678%22%2C%22partner_217%22&applovin_mode=probabilistic&ironsource_mode=ironsource&dimensions=app%2Cchannel%2Ccampaign_network&format_dates=false&full_data=true&include_attr_dependency=true&metrics=cost%2Cinstalls%2Cgross_profit%2Croas_d0%2Croas_d3%2Croas_d7%2Croas_d14%2Croas_d21%2Croas_d28%2Croas_d30%2Croas_d45%2Croas_d50&readable_names=false&sort=-cost&parent_report_id=213219&cost__gt__column=0&is_report_setup_open=true&table_view=pivot"
    st.markdown(f"**Need the report?** Download here: [**Adjust Report Link**]({report_link})")

    uploaded_file = st.file_uploader("Upload CSV:", type=["csv"], key="uploader")

    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            with open(_sid_cache_path(SID), "wb") as f:
                f.write(bytes_data)
            df, roas_cols, roas_days = load_data_from_path_or_file(_sid_cache_path(SID))
            st.session_state.df = df
            st.session_state.roas_columns = roas_cols
            st.session_state.roas_days = roas_days
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.show_analysis_page = True
            st.success("File uploaded & cached to your private session!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    # Clear with confirmation (intro page)
    if st.session_state.df is not None:
        if st.button("Clear File", use_container_width=True, key="clear_open_intro"):
            st.session_state.show_clear_confirm_intro = True

        if st.session_state.get("show_clear_confirm_intro", False):
            st.warning("Are you sure you want to remove the file for this session? This cannot be undone.")
            ci1, ci2 = st.columns(2)
            with ci1:
                if st.button("Yes, remove", key="clear_yes_intro"):
                    st.session_state.df = None
                    st.session_state.roas_columns = None
                    st.session_state.roas_days = None
                    st.session_state.uploaded_filename = None
                    st.session_state.show_analysis_page = False
                    p = _sid_cache_path(SID)
                    if os.path.exists(p): os.remove(p)
                    st.session_state.show_clear_confirm_intro = False
                    st.rerun()
            with ci2:
                if st.button("Cancel", key="clear_no_intro"):
                    st.session_state.show_clear_confirm_intro = False

# -----------------------------
# Analysis Page
# -----------------------------
else:
    st.header("Campaign Performance & Break-Even Analysis")
    st.write("Use the sidebar to choose Platform, Game, Network, and analysis settings.")

    # Sidebar: Clear with confirmation
    st.sidebar.header("Filters")
    if st.sidebar.button("Clear Data", key="clear_open_sidebar"):
        st.session_state.show_clear_confirm = True
    if st.session_state.get("show_clear_confirm", False):
        with st.sidebar:
            st.warning("Are you sure you want to remove the file for this session? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, remove", key="clear_yes_sidebar"):
                    st.session_state.df = None
                    st.session_state.roas_columns = None
                    st.session_state.roas_days = None
                    st.session_state.uploaded_filename = None
                    st.session_state.show_analysis_page = False
                    p = _sid_cache_path(SID)
                    if os.path.exists(p): os.remove(p)
                    st.session_state.show_clear_confirm = False
                    st.rerun()
            with c2:
                if st.button("Cancel", key="clear_no_sidebar"):
                    st.session_state.show_clear_confirm = False

    # Platform filter
    platform_choice = st.sidebar.radio(
        "Platform:",
        ("All", "Android", "iOS"),
        index=0,
        help="Platform is auto-detected from app name (PS_IOS_*, PS_Android_*)."
    )

    base_df = st.session_state.df.copy()
    if platform_choice != "All":
        base_df = base_df[base_df["app_platform"] == platform_choice]

    # Game select (type to search)
    games = sorted(base_df["app_game"].dropna().unique().tolist())
    if not games:
        st.warning("No games found for this platform. Try 'All' or another platform.")
        st.stop()
    display_map = {g: g.replace("_", " ") for g in games}
    selected_game = st.sidebar.selectbox(
        "Select App (Game):",
        games,
        key="selected_game",
        format_func=lambda g: display_map.get(g, g),
        placeholder="Type to search…"
    )

    # Narrow to selected game
    app_df = base_df[base_df["app_game"] == selected_game]

    # Network select
    networks = sorted(app_df["channel"].dropna().unique().tolist())
    if not networks:
        st.warning("No networks found for this game.")
        st.stop()
    selected_network = st.sidebar.selectbox("Select Network:", networks)

    # Exclude last N days
    roas_columns = st.session_state.roas_columns
    max_excludable = max(0, len(roas_columns) - 2)  # keep at least D0 + 1 more day
    days_to_exclude = st.sidebar.number_input(
        "Exclude Last N Days:",
        min_value=0, max_value=max_excludable, value=0, step=1,
        help="Exclude the latest N days if they are unreliable."
    )
    roas_columns_filtered = roas_columns[:-days_to_exclude] if days_to_exclude > 0 else roas_columns[:]

    # Day ints for plotting/logic
    roas_days_filtered = []
    for c in roas_columns_filtered:
        cl = c.lower()
        if cl == "roas_d0": continue
        try: roas_days_filtered.append(int(cl.replace("roas_d","")))
        except: pass
    roas_days_filtered = sorted(set(roas_days_filtered))

    if len(roas_columns_filtered) < 2:
        st.warning("Too many days excluded — need at least D0 and one more day.")
        st.stop()

    # Campaign list — spend-desc order
    chan_df = app_df[app_df["channel"] == selected_network]
    SPEND_COL = "cost"  # change to "spend" if that's your column name
    try:
        spend_by_campaign = (
            chan_df.groupby("campaign_network", dropna=False)[SPEND_COL]
                  .sum(min_count=1)
                  .fillna(0)
                  .sort_values(ascending=False)
        )
        all_campaigns_in_network = list(spend_by_campaign.index)
    except KeyError:
        # Fallback if spend column is missing
        all_campaigns_in_network = sorted(
            chan_df["campaign_network"].dropna().unique().tolist()
        )

    default_list = all_campaigns_in_network[:5] if len(all_campaigns_in_network) > 0 else []
    selected_campaigns = st.multiselect("Select Campaigns:", all_campaigns_in_network, default=default_list)

    # Tabs
    tab1, tab2 = st.tabs(["Standard Analysis", "Scenario Analysis"])

    with tab1:
        st.subheader("Standard Break-Even Analysis")
        st.write("Two modes: (1) Required D0 to break even by a specific day, (2) Predicted day to reach a target ROAS.")
        st.markdown("---")
        st.markdown("#### Standard Analysis Settings")

        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ("Required D0 ROAS for a Day", "Predicted Day for a Target ROAS")
        )

        if analysis_mode == "Required D0 ROAS for a Day":
            c1, c2 = st.columns(2)
            with c1:
                if roas_days_filtered:
                    break_even_day = st.selectbox("Break-Even Day:", roas_days_filtered)
                else:
                    st.warning("No ROAS day columns beyond D0 were found.")
                    st.stop()
            with c2:
                margin_of_error_percent = st.slider("Margin of Error (%)", 0.0, 10.0, 5.0)
            target_roas = 1.00  # fixed 100%
        else:
            target_roas_percent = st.number_input("Target ROAS (%) for Prediction:", 0.0, 200.0, 100.0, 5.0)
            target_roas = target_roas_percent / 100.0
            margin_of_error_percent = 0.0

        st.markdown("---")

        if selected_campaigns:
            # 1) Actual ROAS Values
            st.markdown("### 1) Actual ROAS Values")
            actual_rows = []
            for cname in selected_campaigns:
                cdf = app_df[app_df["campaign_network"] == cname]
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                row = {"Campaign": cname}
                for col in roas_columns_filtered:
                    row[f'ROAS {col.replace("roas_d","D").upper()}'] = means[col]
                actual_rows.append(row)
            if actual_rows:
                adf = pd.DataFrame(actual_rows).set_index("Campaign")
                st.dataframe(
                    adf.style.background_gradient(cmap="YlGnBu", axis=1)
                       .format(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"),
                    use_container_width=True
                )
            else:
                st.info("No ROAS data for selected campaigns.")

            # 2) Cumulative ROAS Growth from D0
            st.markdown("---"); st.markdown("### 2) Cumulative ROAS Growth from D0")
            growth_rows, growth_cols = [], [c for c in roas_columns_filtered if c.lower() != "roas_d0"]
            for cname in selected_campaigns:
                cdf = app_df[app_df["campaign_network"] == cname]
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                d0 = means.get("roas_d0", np.nan)
                if pd.notna(d0) and d0 > 0:
                    g = {"Campaign": cname}
                    for col in growth_cols:
                        g[f'Growth {col.replace("roas_d","D").upper()}'] = (means[col]/d0) - 1
                    growth_rows.append(g)
            if growth_rows:
                gdf = pd.DataFrame(growth_rows).set_index("Campaign")
                st.dataframe(
                    gdf.style.background_gradient(cmap="YlGnBu", axis=1).format("{:.2f}"),
                    use_container_width=True
                )
            else:
                st.info("D0 is zero/NA or no data to compute growth.")

            # 3) Day-over-Day ROAS Growth
            st.markdown("---"); st.markdown("### 3) Day-over-Day ROAS Growth")
            dod_rows, intervals = [], roas_columns_filtered
            for cname in selected_campaigns:
                cdf = app_df[app_df["campaign_network"] == cname]
                means = cdf[intervals].mean(numeric_only=True)
                row = {"Campaign": cname}
                if len(intervals) > 1:
                    for i in range(1, len(intervals)):
                        pcol, ccol = intervals[i-1], intervals[i]
                        prev_v, curr_v = means.get(pcol), means.get(ccol)
                        if pd.notna(prev_v) and prev_v > 0 and pd.notna(curr_v):
                            row[f'Growth {pcol.replace("roas_d","D").upper()}-{ccol.replace("roas_d","D").upper()}'] = (curr_v - prev_v)/prev_v
                        else:
                            row[f'Growth {pcol.replace("roas_d","D").upper()}-{ccol.replace("roas_d","D").upper()}'] = np.nan
                dod_rows.append(row)
            if dod_rows:
                ddf = pd.DataFrame(dod_rows).set_index("Campaign")
                st.dataframe(
                    ddf.style.background_gradient(cmap="YlGnBu", axis=1).format("{:.2f}"),
                    use_container_width=True
                )
            else:
                st.info("Could not compute day-over-day growth.")

            # 4) Individual Campaign Break-Even Analysis
            st.markdown("---"); st.markdown("### 4) Individual Campaign Break-Even Analysis")
            for cname in selected_campaigns:
                st.markdown("---"); st.subheader(f"Campaign: {cname}")
                cdf = app_df[app_df["campaign_network"] == cname]
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                d0 = means.get("roas_d0", np.nan)

                if pd.notna(d0) and d0 > 0:
                    try:
                        days, vals = [], []
                        for col in roas_columns_filtered:
                            dstr = col.lower().replace("roas_d","")
                            if dstr.isdigit():
                                days.append(int(dstr)); vals.append(means[col])
                        if "roas_d0" in roas_columns_filtered and 0 not in days:
                            days = [0] + days; vals = [d0] + vals

                        if analysis_mode == "Required D0 ROAS for a Day":
                            key_col = f"roas_d{break_even_day}"
                            d_target_day = means.get(key_col, np.nan)
                            if pd.notna(d_target_day) and d0 > 0:
                                gm = d_target_day/d0 if d0 != 0 else np.nan
                                if pd.notna(gm) and gm > 0:
                                    required_d0 = 1.00/gm
                                    st.write(f"**Break-Even Day: D{break_even_day}**")
                                    st.write(f"Required D0 ROAS: `{required_d0*100:.2f}%`")
                                    st.write(f"Actual D0 ROAS: `{d0*100:.2f}%`")
                                    diff = d0 - required_d0
                                    if d0 >= required_d0:
                                        st.success(f"On track — **+{abs(diff)*100:.2f}%** above required.")
                                    elif d0 >= required_d0*(1 - margin_of_error_percent/100):
                                        st.warning(f"Within margin ({margin_of_error_percent}%): **-{abs(diff)*100:.2f}%**.")
                                    else:
                                        st.error(f"Below requirement — **-{abs(diff)*100:.2f}%**.")
                            else:
                                st.info(f"D{break_even_day} ROAS column missing or NaN.")

                        else:
                            target_roas = target_roas  # already defined above in this branch
                            be_day = predict_cross_day(days, vals, target=target_roas)
                            if be_day is not None and be_day <= max(days):
                                st.success(f"Predicted to reach **{target_roas*100:.0f}% ROAS** around **Day {be_day:.1f}**.")
                            else:
                                st.error(f"Not predicted to reach **{target_roas*100:.0f}% ROAS** within Day {max(days)}.")

                        # Compact daily table + optional chart
                        sorter = sorted(zip(days, vals), key=lambda t: t[0])
                        cols = [f"Day {d}" for d,_ in sorter]
                        vals_row = [v for _,v in sorter]
                        sdf = pd.DataFrame([vals_row], columns=cols, index=["Average ROAS"])
                        st.dataframe(
                            sdf.style.format(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"),
                            use_container_width=True
                        )

                        show_chart = st.checkbox(
                            f"Show Break-Even Chart for {cname}",
                            value=False, key=f"show_chart_standard_{cname}"
                        )
                        if show_chart:
                            fig, ax = plt.subplots(figsize=(12,7))
                            ax.plot([d for d,_ in sorter], [v for _,v in sorter], marker='o', label='Actual ROAS')
                            ax.axhline(y=1.0, linestyle='-', label='Break-Even (100%)')
                            if analysis_mode == "Predicted Day for a Target ROAS":
                                ax.axhline(y=target_roas, linestyle='--', label=f"Target ({target_roas*100:.0f}%)")
                                if be_day is not None and be_day <= max(days):
                                    ax.axvline(x=be_day, linestyle=':', label=f"Predicted ~{be_day:.1f}")
                            ax.set_title(f'ROAS Performance — {cname}')
                            ax.set_xlabel('Days After Installation'); ax.set_ylabel('Average ROAS')
                            ax.grid(True); ax.legend(); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                            st.pyplot(fig); plt.close(fig)
                    except KeyError:
                        st.error("Columns missing for analysis.")
                else:
                    st.warning("D0 ROAS is zero/NaN — cannot compute growth for this campaign.")
        else:
            st.info("Please select one or more campaigns to analyze.")

    with tab2:
        st.subheader("Break-Even Scenario Analysis")
        st.write("Project break-even day using optimistic and pessimistic multipliers.")
        st.markdown("---")
        st.markdown("#### Scenario Growth Multipliers")
        c1, c2 = st.columns(2)
        with c1:
            optimistic_growth_percent = st.number_input("Optimistic Growth Multiplier (%)", 0, 200, 15, 5)
        with c2:
            pessimistic_growth_percent = st.number_input("Pessimistic Growth Multiplier (%)", 0, 200, 5, 5)
        st.markdown("---")

        if selected_campaigns:
            for cname in selected_campaigns:
                st.markdown("---"); st.subheader(f"Campaign: {cname}")
                cdf = app_df[app_df["campaign_network"] == cname]
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                d0 = means.get("roas_d0", np.nan)
                if pd.notna(d0) and d0 > 0:
                    try:
                        days, base_vals = [], []
                        for col in roas_columns_filtered:
                            dstr = col.lower().replace("roas_d","")
                            if dstr.isdigit():
                                days.append(int(dstr)); base_vals.append(means[col])
                        if "roas_d0" in roas_columns_filtered and 0 not in days:
                            days = [0] + days; base_vals = [d0] + base_vals

                        base_vals = np.array(base_vals, dtype=float)
                        growth_rates = base_vals / d0  # 1.0 at D0

                        opt_mult = 1 + (optimistic_growth_percent/100.0)
                        optimistic_vals = d0 * (1 + (growth_rates - 1) * opt_mult)

                        pes_mult = 1 - (pessimistic_growth_percent/100.0)
                        pessimistic_vals = d0 * (1 + (growth_rates - 1) * pes_mult)

                        opt_be = predict_cross_day(days, optimistic_vals, target=1.0)
                        base_be = predict_cross_day(days, base_vals,       target=1.0)
                        pes_be = predict_cross_day(days, pessimistic_vals, target=1.0)

                        st.write("**Predicted Break-Even Days:**")
                        k1, k2, k3 = st.columns(3)
                        with k1:
                            if opt_be is not None:
                                st.success(f"Optimistic: Day {opt_be:.1f}")
                            else:
                                st.error("Optimistic: Not within timeframe")

                        with k1:
                            if base_be is not None:
                                st.info(f"Base: Day {base_be:.1f}")
                            else:
                                st.error("Base: Not within timeframe")

                        with k3:
                            if pes_be is not None:
                                st.warning(f"Pessimistic: Day {pes_be:.1f}")
                            else:
                                st.warning(f"Pessimistic: Day {pes_be:.1f}")

                        show_chart = st.checkbox(
                            f"Show Scenario Chart for {cname}",
                            value=False, key=f"show_chart_scenario_{cname}"
                        )
                        if show_chart:
                            sorter = sorted(zip(days, optimistic_vals, base_vals, pessimistic_vals), key=lambda t: t[0])
                            xs = [t[0] for t in sorter]
                            ys_opt = [t[1] for t in sorter]
                            ys_base = [t[2] for t in sorter]
                            ys_pes = [t[3] for t in sorter]

                            fig, ax = plt.subplots(figsize=(12,7))
                            ax.plot(xs, ys_opt, marker='o', linestyle='--', label=f'Optimistic (+{optimistic_growth_percent}%)')
                            ax.plot(xs, ys_base, marker='o', label='Base (Historical)')
                            ax.plot(xs, ys_pes, marker='o', linestyle='--', label=f'Pessimistic (-{pessimistic_growth_percent}%)')
                            ax.axhline(y=1.0, linestyle='-', label='Break-Even (100%)')

                            if opt_be is not None: ax.axvline(x=opt_be, linestyle=':', label=f'Opt. ~{opt_be:.1f}')
                            if base_be is not None: ax.axvline(x=base_be, linestyle=':', label=f'Base ~{base_be:.1f}')
                            if pes_be is not None: ax.axvline(x=pes_be, linestyle=':', label=f'Pess. ~{pes_be:.1f}')

                            ax.set_title(f'Break-Even Scenarios — {cname}')
                            ax.set_xlabel('Days After Installation'); ax.set_ylabel('Average ROAS')
                            ax.grid(True); ax.legend(); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                            st.pyplot(fig); plt.close(fig)
                    except KeyError:
                        st.error("Columns missing for scenario analysis.")
                else:
                    st.warning("D0 ROAS is zero/NaN — cannot compute scenario growth.")
        else:
            st.info("Please select one or more campaigns to analyze.")





