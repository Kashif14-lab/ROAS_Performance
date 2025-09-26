# ROAS Analyzer — Full App with Weekly Controls & Weekly Analysis
# Run: streamlit run app.py

import os
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from datetime import date

# =============== Page config ===============
st.set_page_config(layout="wide")
st.markdown("""
<style>
.stTabs [data-testid="stTab"] button { font-size: 1.1rem; font-weight: 700; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { margin-top: .4rem; }
</style>
""", unsafe_allow_html=True)

# =============== Per-user cache via ?sid= ===============
def get_or_set_sid():
    if "sid" in st.query_params and st.query_params["sid"]:
        return st.query_params["sid"]
    sid = str(uuid.uuid4())
    st.query_params["sid"] = sid
    return sid

SID = get_or_set_sid()
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)
def _sid_cache_path(sid: str) -> str:
    return os.path.join(DATA_DIR, f"{sid}_last_upload.csv")

# =============== Helpers ===============
def _parse_roas_series(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('%','').replace(',','').strip()
    return pd.to_numeric(s, errors='coerce') / 100.0

def parse_app_fields(app_str):
    s = str(app_str) if pd.notna(app_str) else ""
    parts = s.split("_")
    platform, game = "Unknown", s
    if len(parts) >= 3:
        plat_raw = parts[1].lower()
        if "ios" in plat_raw: platform = "iOS"
        elif "and" in plat_raw or "android" in plat_raw: platform = "Android"
        else: platform = parts[1]
        game = "_".join(parts[2:])
    return pd.Series({"app_platform": platform, "app_game": game})

def _is_num(x):
    return x is not None and isinstance(x, (int, float, np.floating)) and np.isfinite(x)

def predict_cross_day(days, values, target=1.0):
    if not days or values is None:
        return None
    x = np.array(days, dtype=float)
    y = pd.Series(values, dtype="float64")
    if y.isna().all(): return None
    y = y.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    y = np.maximum.accumulate(y.values)
    if y[0] >= target: return 0.0
    if np.nanmax(y) < target: return None
    idx = int(np.argmax(y >= target))
    if idx == 0: return 0.0
    x1, x2 = x[idx-1], x[idx]; y1, y2 = y[idx-1], y[idx]
    if pd.isna(y1) or pd.isna(y2): return None
    if y2 == y1: return x2
    return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

def _normalize_roas_name(c: str) -> str:
    cl = str(c).strip().lower().replace(" ", "").replace("-", "").replace("__", "_")
    if cl.startswith("roasd"):
        rest = cl.replace("roasd", "")
        if rest.isdigit():
            return f"roas_d{rest}"
    if cl.startswith("roas_d"):
        return cl.replace("roas_d_", "roas_d")
    if cl.startswith("roas") and "d" in cl:
        try:
            num = cl.split("d", 1)[1]
            if num.isdigit():
                return f"roas_d{num}"
        except Exception:
            pass
    return c

def _running_median(arr: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or len(arr) < k:
        return arr
    out = arr.copy()
    pad = k // 2
    for i in range(len(arr)):
        s = max(0, i - pad); e = min(len(arr), i + pad + 1)
        out[i] = np.nanmedian(arr[s:e])
    return out

def _interp_extrap_from_curve(mult_curve: pd.Series, target_day: int) -> float:
    day_idx, mult_vals = [], []
    for name, val in mult_curve.items():
        cl = str(name).lower()
        if cl.startswith("roas_d"):
            d = cl.replace("roas_d","")
            if d.isdigit():
                day_idx.append(int(d)); mult_vals.append(float(val))
    if not day_idx:
        return np.nan
    order = np.argsort(day_idx)
    xs = np.array(day_idx)[order]; ys = np.array(mult_vals)[order]
    if target_day <= xs.min(): return ys[0]
    if target_day >= xs.max():
        if len(xs) >= 3:
            x1,x2,x3 = xs[-3], xs[-2], xs[-1]
            y1,y2,y3 = ys[-3], ys[-2], ys[-1]
            m1 = (y3 - y2) / (x3 - x2) if x3 != x2 else 0.0
            m0 = (y2 - y1) / (x2 - x1) if x2 != x1 else m1
            m  = 0.5*(m0 + m1)
            return max(y3 + (target_day - x3)*m, y3)
        elif len(xs) >= 2:
            m = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return max(ys[-1] + (target_day - xs[-1])*m, ys[-1])
        else:
            return ys[-1]
    return float(np.interp(target_day, xs, ys))

# ----- Pooled robust median multipliers (fallback / non-weekly) -----
def pooled_multiplier_curve(df_block: pd.DataFrame, roas_cols: list[str],
                            min_rows: int = 3, tiny_delta: float = 1e-3,
                            stale_run: int = 4) -> pd.Series:
    if df_block is None or df_block.empty:
        return pd.Series(dtype=float)
    low_cols = [str(c).lower() for c in roas_cols]
    if "roas_d0" not in low_cols: return pd.Series(dtype=float)
    col_map = {str(c).lower(): c for c in roas_cols}
    d0col = col_map["roas_d0"]

    d0 = df_block[d0col].replace(0, np.nan)
    ratios = pd.DataFrame(index=df_block.index)
    for c in roas_cols:
        ratios[c] = df_block[c] / d0

    contrib = ratios.notna().sum(axis=0)
    med = ratios.median(axis=0, numeric_only=True).replace([np.inf, -np.inf], np.nan)
    med = med.interpolate(limit_direction="both")
    med = pd.Series(np.maximum.accumulate(med.values), index=med.index)
    for name in med.index:
        if str(name).lower()=="roas_d0":
            med.loc[name]=1.0; break

    # to arrays
    days, vals = [], []
    for col in med.index:
        cl = str(col).lower()
        if cl.startswith("roas_d"):
            dd = cl.replace("roas_d","")
            if dd.isdigit():
                days.append(int(dd)); vals.append(med[col])
    if not days: return pd.Series(dtype=float)
    order = np.argsort(days)
    xs = np.array(days)[order]; ys = np.array(vals)[order]

    # tiny-delta stale tail trim
    deltas = np.diff(ys)
    tiny = np.where(deltas < tiny_delta)[0]
    if len(tiny) >= stale_run:
        tail_count = 0
        for i in range(len(deltas)-1, -1, -1):
            if deltas[i] < tiny_delta: tail_count += 1
            else: break
        if tail_count >= stale_run:
            cut_after = int(xs[-(tail_count+1)])
            mask_good = xs <= cut_after
            xs = xs[mask_good]; ys = ys[mask_good]

    cleaned = pd.Series(index=[f"roas_d{int(d)}" for d in xs], data=ys, dtype=float)
    cleaned = cleaned.interpolate(limit_direction="both")
    cleaned.loc[:] = np.maximum.accumulate(cleaned.values)
    cleaned.loc[:] = _running_median(cleaned.values.astype(float), k=3)
    if "roas_d0" in cleaned.index: cleaned.loc["roas_d0"]=1.0
    return cleaned

# ----- Weighted median helper -----
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights>0)
    if not mask.any(): return np.nan
    v = values[mask]; w = weights[mask]
    order = np.argsort(v)
    v = v[order]; w = w[order]
    cw = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cw, 0.5)
    idx = min(idx, len(v)-1)
    return float(v[idx])

# ----- WEEKLY: build weekly-normalized weighted curve -----
def weekly_multiplier_curve(df_block: pd.DataFrame, roas_cols: list[str],
                            week_col: str, weeks_selected: list,
                            weight_by: str = "cost",
                            half_life_weeks: float = 0.0) -> (pd.Series, pd.DataFrame):
    """
    Returns:
      - cleaned multiplier curve (Series indexed by roas_d*)
      - debug dataframe of selected weeks with weights
    """
    if df_block is None or df_block.empty or week_col not in df_block.columns:
        return pd.Series(dtype=float), pd.DataFrame()

    # map ROAS columns case-insensitively
    col_map = {str(c).lower(): c for c in roas_cols}
    if "roas_d0" not in col_map: return pd.Series(dtype=float), pd.DataFrame()
    d0col = col_map["roas_d0"]

    # choose weight column
    SPEND_COL = "cost"  # rename if your sheet uses "spend"
    weight_col = None
    if weight_by.lower().startswith("inst") and "installs" in df_block.columns:
        weight_col = "installs"
    elif SPEND_COL in df_block.columns:
        weight_col = SPEND_COL
    elif "installs" in df_block.columns:
        weight_col = "installs"

    # 1) per-row ratios
    ratios = df_block.copy()
    d0 = ratios[d0col].replace(0, np.nan)
    for c in roas_cols:
        ratios[c] = ratios[c] / d0

    # 2) per-week median ratios
    use = ratios[ratios[week_col].isin(weeks_selected)].copy()
    if use.empty: return pd.Series(dtype=float), pd.DataFrame()
    week_groups = use.groupby(week_col, dropna=False)
    week_median = week_groups[roas_cols].median(numeric_only=True)

    # 2b) base weights per week
    if weight_col and weight_col in df_block.columns:
        base_w = df_block[df_block[week_col].isin(weeks_selected)].groupby(week_col)[weight_col].sum(min_count=1)
    else:
        base_w = pd.Series(1.0, index=week_median.index)

    # 2c) freshness decay
    week_idx_sorted = sorted(weeks_selected)
    if len(week_idx_sorted) == 0:
        return pd.Series(dtype=float), pd.DataFrame()
    max_week = max(week_idx_sorted)

    def _age_in_weeks(w):
        if isinstance(max_week, (pd.Timestamp, date)) and isinstance(w, (pd.Timestamp, date)):
            days = (max_week - w).days
            return float(days) / 7.0
        return 0.0

    ages = pd.Series({_w: _age_in_weeks(_w) for _w in week_median.index})

    if half_life_weeks and half_life_weeks > 0:
        lam = np.log(2) / half_life_weeks
        decay = np.exp(-lam * ages)
    else:
        decay = pd.Series(1.0, index=week_median.index)

    eff_w = (base_w.reindex(week_median.index).fillna(0) * decay).astype(float)

    # 3) weighted median per ROAS day across weeks
    day_vals = {}
    for c in roas_cols:
        vals = week_median[c].values.astype(float)
        wts  = eff_w.values.astype(float)
        day_vals[c] = weighted_median(vals, wts)

    mult = pd.Series(day_vals)
    mult = mult.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    mult.loc[:] = np.maximum.accumulate(mult.values)
    mult.loc[:] = _running_median(mult.values.astype(float), k=3)
    for name in mult.index:
        if str(name).lower()=="roas_d0":
            mult.loc[name]=1.0; break

    dbg = pd.DataFrame({
        "week": week_median.index,
        "base_weight": base_w.reindex(week_median.index).fillna(0).values,
        "age_weeks": ages.values,
        "decay": decay.values,
        "effective_weight": eff_w.values
    })
    return mult, dbg

# =============== Data load & preprocess ===============
@st.cache_data
def load_data_from_path_or_file(path_or_file):
    if isinstance(path_or_file, str):
        df = pd.read_csv(path_or_file, low_memory=False)
    else:
        df = pd.read_csv(path_or_file, low_memory=False)

    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = [_normalize_roas_name(c) for c in df.columns]

    required = {"app", "channel", "campaign_network"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    parsed = df["app"].apply(parse_app_fields)
    df["app_platform"] = parsed["app_platform"]
    df["app_game"] = parsed["app_game"]

    roas_columns = [c for c in df.columns if str(c).lower().startswith("roas_d")]
    if not roas_columns:
        raise ValueError("No ROAS columns found (expected roas_d0, roas_d3, ...).")
    for c in roas_columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(_parse_roas_series)

    df["max_roas"] = df[roas_columns].max(axis=1)

    roas_days = []
    for c in roas_columns:
        cl = str(c).lower()
        if cl == "roas_d0": continue
        d = cl.replace("roas_d","")
        if d.isdigit(): roas_days.append(int(d))
    roas_days = sorted(set(roas_days))

    # weekly parse if present
    wk_candidates = [c for c in df.columns if c.lower()=="week"]
    if wk_candidates:
        wcol = wk_candidates[0]
        try:
            df[wcol] = pd.to_datetime(df[wcol], errors="coerce").dt.date
        except Exception:
            pass

    return df, roas_columns, roas_days

# =============== Session init & auto-reload ===============
if "show_analysis_page" not in st.session_state:
    st.session_state.show_analysis_page = False
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.roas_columns = None
    st.session_state.roas_days = None
    st.session_state.uploaded_filename = None

if st.session_state.df is not None:
    st.session_state.show_analysis_page = True

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

# =============== Intro / Upload ===============
if not st.session_state.show_analysis_page:
    st.title("Campaign Performance Analyzer (ROAS)")
    st.write("Upload an Adjust CSV to analyze ROAS; your file persists privately to your session.")

    uploaded_file = st.file_uploader("Upload CSV:", type=["csv"], key="uploader")
    if uploaded_file is not None:
        try:
            with open(_sid_cache_path(SID), "wb") as f:
                f.write(uploaded_file.getvalue())
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

# =============== Analysis Page ===============
else:
    st.header("Campaign Performance & Break-Even Analysis")

    # Sidebar: clear confirm
    st.sidebar.header("Filters")
    if st.sidebar.button("Clear Data", key="clear_open_sidebar"):
        st.session_state.show_clear_confirm = True
    if st.session_state.get("show_clear_confirm", False):
        with st.sidebar:
            st.warning("Remove the cached file for this session?")
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

    # Platform → Game → Network
    platform_choice = st.sidebar.radio("Platform:", ("All", "Android", "iOS"), index=0)
    base_df = st.session_state.df.copy()
    if platform_choice != "All":
        base_df = base_df[base_df["app_platform"] == platform_choice]

    games = sorted(base_df["app_game"].dropna().unique().tolist())
    if not games:
        st.warning("No games found for this platform."); st.stop()
    display_map = {g: g.replace("_", " ") for g in games}
    selected_game = st.sidebar.selectbox(
        "Select App (Game):", games, key="selected_game",
        format_func=lambda g: display_map.get(g, g), placeholder="Type to search…"
    )
    app_df = base_df[base_df["app_game"] == selected_game]

    networks = sorted(app_df["channel"].dropna().unique().tolist())
    if not networks:
        st.warning("No networks found for this game."); st.stop()
    selected_network = st.sidebar.selectbox("Select Network:", networks)

    # Exclude last N days (for tables/analyses)
    roas_columns = st.session_state.roas_columns
    max_excludable = max(0, len(roas_columns) - 2)
    days_to_exclude = st.sidebar.number_input(
        "Exclude Last N Days:", min_value=0, max_value=max_excludable, value=0, step=1
    )
    roas_columns_filtered = roas_columns[:-days_to_exclude] if days_to_exclude > 0 else roas_columns[:]
    roas_days_filtered = sorted({int(str(c).lower().replace("roas_d","")) for c in roas_columns_filtered if str(c).lower()!="roas_d0" and str(c).lower().startswith("roas_d")})
    if len(roas_columns_filtered) < 2:
        st.warning("Too many days excluded — need at least D0 and one more day."); st.stop()

    # Campaigns sorted by spend (use view_df later after weekly toggle)
    chan_df_base = app_df[app_df["channel"] == selected_network]
    SPEND_COL = "cost"  # change if your sheet uses a different spend column
    try:
        spend_by_campaign = (
            chan_df_base.groupby("campaign_network", dropna=False)[SPEND_COL]
                  .sum(min_count=1).fillna(0).sort_values(ascending=False)
        )
        all_campaigns_in_network = list(spend_by_campaign.index)
    except KeyError:
        all_campaigns_in_network = sorted(chan_df_base["campaign_network"].dropna().unique().tolist())
    default_list = all_campaigns_in_network[:5]
    selected_campaigns = st.multiselect("Select Campaigns:", all_campaigns_in_network, default=default_list)

    # ========== Weekly controls (drive view + projections) ==========
    has_weekly = "week" in [c.lower() for c in app_df.columns]
    wcol = next((c for c in app_df.columns if c.lower()=="week"), None)
    view_df = app_df.copy()  # this drives the three standard tables when toggle ON

    weeks_selected = []
    weight_by = "Spend"; half_life_weeks = 6; exclude_incomplete = True
    if has_weekly:
        st.markdown("#### Weekly settings")
        block_net = app_df[app_df["channel"] == selected_network].copy()
        weeks_all = sorted(block_net[wcol].dropna().unique().tolist())

        exclude_incomplete = st.checkbox("Exclude latest (possibly incomplete) week", value=True, key="wk_excl")
        if exclude_incomplete and len(weeks_all) >= 1:
            weeks_for_default = weeks_all[:-1]
        else:
            weeks_for_default = weeks_all[:]

        default_weeks = weeks_for_default[-min(13, len(weeks_for_default)):] if weeks_for_default else weeks_all[-min(13, len(weeks_all)):]
        weeks_selected = st.multiselect("Include Weeks:", weeks_all, default=default_weeks, key="weeks_select_main")

        use_weekly_for_tables = st.toggle("Show Standard tables using selected weeks", value=True, help="If ON, Actual/Cumulative/DoD tables use only the selected weeks. Break-even uses overall pooled data.")
        weight_by = st.radio("Weight by (for Weekly normalized curve):", ("Spend", "Installs"), horizontal=True, key="wk_weight_by")
        half_life_weeks = st.slider("Freshness decay half-life (weeks) for Weekly curve", min_value=0, max_value=12, value=6, help="0 = no decay; higher = recent weeks get more weight", key="wk_halflife")

        if use_weekly_for_tables and weeks_selected:
            view_df = app_df[app_df[wcol].isin(weeks_selected)].copy()

    # Campaign spend order should reflect the table view if weekly is ON
    try:
        spend_by_campaign_view = (
            view_df[view_df["channel"]==selected_network].groupby("campaign_network", dropna=False)[SPEND_COL]
                .sum(min_count=1).fillna(0).sort_values(ascending=False)
        )
        sorted_campaigns_view = list(spend_by_campaign_view.index)
        # re-order selected defaults (UI already chosen, so skip altering selection)
    except Exception:
        pass

    # ========== Tabs ==========
    tab1, tab2, tab3 = st.tabs(["Standard Analysis", "Scenario Analysis", "Weekly Analysis"])

    # --------------------- TAB 1 ---------------------
    with tab1:
        st.subheader("Standard Break-Even Analysis")
        st.write("Use the settings below to compute Required D0 or Predicted Day to target ROAS.")
        st.markdown("---")
        st.markdown("#### Standard Analysis Settings")
        analysis_mode = st.radio("Select Analysis Mode:", ("Required D0 ROAS for a Day", "Predicted Day for a Target ROAS"))
        if analysis_mode == "Required D0 ROAS for a Day":
            c1, c2 = st.columns(2)
            with c1:
                if roas_days_filtered:
                    break_even_day = st.selectbox("Break-Even Day:", roas_days_filtered)
                else:
                    st.warning("No ROAS day columns beyond D0 found."); st.stop()
            with c2:
                margin_of_error_percent = st.slider("Margin of Error (%)", 0.0, 10.0, 5.0)
            target_roas = 1.00
        else:
            target_roas_percent = st.number_input("Target ROAS (%) for Prediction:", 0.0, 200.0, 100.0, 5.0)
            target_roas = target_roas_percent / 100.0
            margin_of_error_percent = 0.0

        st.markdown("---")

        # ======== Quick Projection (Weekly-aware) ========
        st.markdown("### Quick Projection from D0")

        # D0 input (auto from last selected week if weekly exists)
        if has_weekly and weeks_selected:
            last_sel_week = max(weeks_selected)
            d0col = next((c for c in roas_columns_filtered if str(c).lower()=="roas_d0"), None)
            if d0col is not None:
                d0_last_week = app_df[(app_df[wcol]==last_sel_week) & (app_df["channel"]==selected_network)][d0col].mean()
                if pd.notna(d0_last_week):
                    proj_d0 = d0_last_week
                    st.info(f"Auto-filled D0 from last selected week ({last_sel_week}): **{proj_d0*100:.2f}%**")
                else:
                    proj_d0 = st.number_input("Enter D0 ROAS (%)", 0.0, 500.0, 20.0) / 100.0
            else:
                proj_d0 = st.number_input("Enter D0 ROAS (%)", 0.0, 500.0, 20.0) / 100.0
        else:
            proj_d0 = st.number_input("Enter D0 ROAS (%)", 0.0, 500.0, 20.0) / 100.0

        # Baseline choice
        baseline_label = "Baseline growth curve:"
        options = ["Selected campaign (avg of selected)", "Network-level median (this game + network)"]
        if has_weekly:
            options.append("Weekly normalized (recommended)")
        baseline_source = st.radio(baseline_label, options, horizontal=True)

        # Build baseline multiplier curve
        mult_curve = None
        dbg_weeks = pd.DataFrame()

        if baseline_source.startswith("Selected"):
            rows = []
            for cname in selected_campaigns:
                cdf = view_df[view_df["campaign_network"] == cname]  # weekly filter applied if ON
                if not cdf.empty: rows.append(cdf)
            block = pd.concat(rows) if rows else pd.DataFrame()
            mult_curve = pooled_multiplier_curve(block, roas_columns_filtered, min_rows=3)

        elif baseline_source.startswith("Network"):
            net_df = view_df[view_df["channel"] == selected_network]  # weekly filter applied if ON
            mult_curve = pooled_multiplier_curve(net_df, roas_columns_filtered, min_rows=5)

        else:
            # Weekly normalized with user-selected weeks & weights
            net_df_full = app_df[app_df["channel"] == selected_network]  # use full set for weekly combine
            if has_weekly and weeks_selected:
                mult_curve, dbg_weeks = weekly_multiplier_curve(
                    net_df_full, roas_columns_filtered, wcol, weeks_selected,
                    weight_by=("inst" if weight_by.lower().startswith("inst") else "cost"),
                    half_life_weeks=float(half_life_weeks)
                )
            else:
                st.warning("No weeks selected — falling back to pooled network baseline.")
                mult_curve = pooled_multiplier_curve(net_df_full, roas_columns_filtered, min_rows=5)

        # Debug expander
        with st.expander("Debug: projection inputs", expanded=False):
            st.write("Detected ROAS columns:", roas_columns_filtered)
            if not dbg_weeks.empty:
                st.write("Selected weeks & effective weights:")
                st.dataframe(dbg_weeks.rename(columns={"week":"week"}), use_container_width=True)
            if mult_curve is None or mult_curve.empty:
                st.write("Multiplier curve: **EMPTY**")
            else:
                st.write("Multiplier curve (cleaned/monotonic):")
                st.dataframe(mult_curve.to_frame("mult").style.format("{:.4f}"))
                # show deltas
                try:
                    idx_sorted = sorted([int(str(i).lower().replace("roas_d","")) for i in mult_curve.index])
                    vals = [mult_curve[f"roas_d{d}"] for d in idx_sorted]
                    st.write("Deltas (next - prev):", np.round(np.diff(vals), 6))
                except Exception:
                    pass

        # Projections strip
        if mult_curve is None or mult_curve.empty or mult_curve.nunique(dropna=True) <= 1:
            st.warning("Baseline curve is flat/insufficient after cleaning. Projections skipped.")
        else:
            target_days = [30, 45, 50, 60, 75]
            cols_proj = st.columns(len(target_days))
            for i, d in enumerate(target_days):
                mult_d = _interp_extrap_from_curve(mult_curve, d)
                val = proj_d0 * mult_d if np.isfinite(mult_d) else np.nan
                with cols_proj[i]:
                    st.metric(f"D{d}", f"{val*100:.2f}%" if pd.notna(val) else "—")
        # ======== /Quick Projection ========

        # ---- Standard tables (use view_df if toggle ON) ----
        if selected_campaigns:
            # 1) Actual ROAS
            st.markdown("---"); st.markdown("### 1) Actual ROAS Values")
            actual_rows = []
            for cname in selected_campaigns:
                cdf = view_df[view_df["campaign_network"] == cname]
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                row = {"Campaign": cname}
                for col in roas_columns_filtered:
                    row[f'ROAS {str(col).replace("roas_d","D").upper()}'] = means[col]
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
            growth_rows, growth_cols = [], [c for c in roas_columns_filtered if str(c).lower() != "roas_d0"]
            for cname in selected_campaigns:
                cdf = view_df[view_df["campaign_network"] == cname]
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                d0_key = next((nm for nm in roas_columns_filtered if str(nm).lower()=="roas_d0"), None)
                d0 = means.get(d0_key, np.nan) if d0_key else np.nan
                if pd.notna(d0) and d0 > 0:
                    g = {"Campaign": cname}
                    for col in growth_cols:
                        g[f'Growth {str(col).replace("roas_d","D").upper()}'] = (means[col]/d0) - 1
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
                cdf = view_df[view_df["campaign_network"] == cname]
                means = cdf[intervals].mean(numeric_only=True)
                row = {"Campaign": cname}
                if len(intervals) > 1:
                    for i in range(1, len(intervals)):
                        pcol, ccol = intervals[i-1], intervals[i]
                        prev_v, curr_v = means.get(pcol), means.get(ccol)
                        if pd.notna(prev_v) and prev_v > 0 and pd.notna(curr_v):
                            row[f'Growth {str(pcol).replace("roas_d","D").upper()}-{str(ccol).replace("roas_d","D").upper()}'] = (curr_v - prev_v)/prev_v
                        else:
                            row[f'Growth {str(pcol).replace("roas_d","D").upper()}-{str(ccol).replace("roas_d","D").upper()}'] = np.nan
                dod_rows.append(row)
            if dod_rows:
                ddf = pd.DataFrame(dod_rows).set_index("Campaign")
                st.dataframe(
                    ddf.style.background_gradient(cmap="YlGnBu", axis=1).format("{:.2f}"),
                    use_container_width=True
                )
            else:
                st.info("Could not compute day-over-day growth.")

            # 4) Individual Campaign Break-Even Analysis (OVERALL pooled)
            st.markdown("---"); st.markdown("### 4) Individual Campaign Break-Even Analysis (Overall)")
            for cname in selected_campaigns:
                st.markdown("---"); st.subheader(f"Campaign: {cname}")
                # IMPORTANT: use overall pooled (app_df), not view_df
                cdf = app_df[app_df["campaign_network"] == cname]
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                d0_key = next((nm for nm in roas_columns_filtered if str(nm).lower()=="roas_d0"), None)
                d0 = means.get(d0_key, np.nan) if d0_key else np.nan

                if pd.notna(d0) and d0 > 0:
                    try:
                        days, vals = [], []
                        for col in roas_columns_filtered:
                            cl = str(col).lower()
                            if cl.startswith("roas_d"):
                                d = cl.replace("roas_d","")
                                if d.isdigit():
                                    days.append(int(d)); vals.append(means[col])
                        if d0_key and 0 not in days:
                            days = [0] + days; vals = [d0] + vals

                        if analysis_mode == "Required D0 ROAS for a Day":
                            target_col = f"roas_d{break_even_day}"
                            d_target = means.get(target_col, np.nan)
                            if pd.notna(d_target) and d0 > 0:
                                gm = d_target/d0 if d0 != 0 else np.nan
                                if pd.notna(gm) and gm > 0:
                                    req_d0 = 1.00/gm
                                    st.write(f"**Break-Even Day: D{break_even_day}**")
                                    st.write(f"Required D0 ROAS: `{req_d0*100:.2f}%`")
                                    st.write(f"Actual D0 ROAS: `{d0*100:.2f}%`")
                                    diff = d0 - req_d0
                                    if d0 >= req_d0:
                                        st.success(f"On track — **+{abs(diff)*100:.2f}%** above required.")
                                    elif d0 >= req_d0*(1 - margin_of_error_percent/100):
                                        st.warning(f"Within margin ({margin_of_error_percent}%): **-{abs(diff)*100:.2f}%**.")
                                    else:
                                        st.error(f"Below requirement — **-{abs(diff)*100:.2f}%**.")
                            else:
                                st.info(f"D{break_even_day} ROAS missing/NaN.")
                        else:
                            be_day = predict_cross_day(days, vals, target=target_roas)
                            if _is_num(be_day) and be_day <= max(days):
                                st.success(f"Predicted to reach **{target_roas*100:.0f}% ROAS** around **Day {be_day:.1f}**.")
                            else:
                                st.error(f"Not predicted to reach **{target_roas*100:.0f}% ROAS** within Day {max(days)}.")

                        sorter = sorted(zip(days, vals), key=lambda t: t[0])
                        cols = [f"Day {d}" for d,_ in sorter]
                        vals_row = [v for _,v in sorter]
                        sdf = pd.DataFrame([vals_row], columns=cols, index=["Average ROAS"])
                        st.dataframe(
                            sdf.style.format(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"), use_container_width=True
                        )

                        show_chart = st.checkbox(
                            f"Show Break-Even Chart for {cname}", value=False, key=f"show_chart_standard_{cname}"
                        )
                        if show_chart:
                            fig, ax = plt.subplots(figsize=(12,7))
                            ax.plot([d for d,_ in sorter], [v for _,v in sorter], marker='o', label='Actual ROAS')
                            ax.axhline(y=1.0, linestyle='-', label='Break-Even (100%)')
                            if analysis_mode == "Predicted Day for a Target ROAS":
                                ax.axhline(y=target_roas, linestyle='--', label=f"Target ({target_roas*100:.0f}%)")
                                if _is_num(be_day) and be_day <= max(days):
                                    ax.axvline(x=be_day, linestyle=':', label=f"Pred ~{be_day:.1f}")
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

    # --------------------- TAB 2 ---------------------
    with tab2:
        st.subheader("Break-Even Scenario Analysis")
        st.write("Project break-even day using optimistic and pessimistic multipliers on historical growth.")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            optimistic_growth_percent = st.number_input("Optimistic Growth Multiplier (%)", 0, 200, 15, 5)
        with c2:
            pessimistic_growth_percent = st.number_input("Pessimistic Growth Multiplier (%)", 0, 200, 5, 5)
        st.markdown("---")

        if selected_campaigns:
            for cname in selected_campaigns:
                st.markdown("---"); st.subheader(f"Campaign: {cname}")
                cdf = view_df[view_df["campaign_network"] == cname]  # scenario can use weekly-filtered view
                means = cdf[roas_columns_filtered].mean(numeric_only=True)
                d0_key = next((nm for nm in roas_columns_filtered if str(nm).lower()=="roas_d0"), None)
                d0 = means.get(d0_key, np.nan) if d0_key else np.nan

                if pd.notna(d0) and d0 > 0:
                    try:
                        days, base_vals = [], []
                        for col in roas_columns_filtered:
                            cl = str(col).lower()
                            if cl.startswith("roas_d"):
                                d = cl.replace("roas_d","")
                                if d.isdigit():
                                    days.append(int(d)); base_vals.append(means[col])
                        if d0_key and 0 not in days:
                            days = [0] + days; base_vals = [d0] + base_vals

                        base_vals = np.array(base_vals, dtype=float)
                        growth_rates = base_vals / d0  # 1.0 at D0

                        opt_mult = 1 + (optimistic_growth_percent/100.0)
                        optimistic_vals = d0 * (1 + (growth_rates - 1) * opt_mult)

                        pes_mult = 1 - (pessimistic_growth_percent/100.0)
                        pessimistic_vals = d0 * (1 + (growth_rates - 1) * pes_mult)

                        opt_be  = predict_cross_day(days, optimistic_vals, target=1.0)
                        base_be = predict_cross_day(days, base_vals,       target=1.0)
                        pes_be  = predict_cross_day(days, pessimistic_vals, target=1.0)

                        st.write("**Predicted Break-Even Days:**")
                        k1, k2, k3 = st.columns(3)
                        with k1:
                            if _is_num(opt_be): st.success(f"Optimistic: Day {opt_be:.1f}")
                            else: st.error("Optimistic: Not within timeframe")
                        with k2:
                            if _is_num(base_be): st.info(f"Base: Day {base_be:.1f}")
                            else: st.error("Base: Not within timeframe")
                        with k3:
                            if _is_num(pes_be): st.warning(f"Pessimistic: Day {pes_be:.1f}")
                            else: st.error("Pessimistic: Not within timeframe")

                        show_chart = st.checkbox(
                            f"Show Scenario Chart for {cname}",
                            value=False, key=f"show_chart_scenario_{cname}"
                        )
                        if show_chart:
                            sorter = sorted(zip(days, optimistic_vals, base_vals, pessimistic_vals), key=lambda t: t[0])
                            xs = [t[0] for t in sorter]
                            ys_opt  = [t[1] for t in sorter]
                            ys_base = [t[2] for t in sorter]
                            ys_pes  = [t[3] for t in sorter]

                            fig, ax = plt.subplots(figsize=(12,7))
                            ax.plot(xs, ys_opt, marker='o', linestyle='--', label=f'Optimistic (+{optimistic_growth_percent}%)')
                            ax.plot(xs, ys_base, marker='o', label='Base (Historical)')
                            ax.plot(xs, ys_pes, marker='o', linestyle='--', label=f'Pessimistic (-{pessimistic_growth_percent}%)')
                            ax.axhline(y=1.0, linestyle='-', label='Break-Even (100%)')

                            if _is_num(opt_be):  ax.axvline(x=opt_be,  linestyle=':', label=f'Opt. ~{opt_be:.1f}')
                            if _is_num(base_be): ax.axvline(x=base_be, linestyle=':', label=f'Base ~{base_be:.1f}')
                            if _is_num(pes_be):  ax.axvline(x=pes_be,  linestyle=':', label=f'Pess. ~{pes_be:.1f}')

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

    # --------------------- TAB 3 ---------------------
    with tab3:
        st.subheader("Weekly Analysis (by Week)")
        if not has_weekly:
            st.warning("`week` column not found in CSV."); st.stop()

        if not weeks_selected:
            st.info("Select at least one week in Weekly settings (above) to view weekly tables.")
            st.stop()

        wk_df = app_df[app_df[wcol].isin(weeks_selected) & (app_df["channel"]==selected_network)]

        if selected_campaigns:
            # A) Weekly Actual ROAS
            st.markdown("### A) Weekly Actual ROAS")
            key_days = [0,3,7,14,21,28,30,45,50,60,75]
            # filter to a reasonable set of columns:
            show_cols = []
            for c in roas_columns_filtered:
                cl = str(c).lower()
                if cl == "roas_d0":
                    show_cols.append(c); continue
                if cl.startswith("roas_d"):
                    d = cl.replace("roas_d","")
                    if d.isdigit() and int(d) in key_days:
                        show_cols.append(c)
            if not show_cols:
                show_cols = roas_columns_filtered

            for wk in sorted(weeks_selected):
                sub = wk_df[(wk_df[wcol]==wk) & (wk_df["campaign_network"].isin(selected_campaigns))]
                if sub.empty:
                    st.info(f"No data for week {wk}")
                    continue
                tbl = sub.groupby("campaign_network")[show_cols].mean(numeric_only=True)
                st.markdown(f"**Week: {wk}**")
                st.dataframe(
                    tbl.style.format(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
                       .background_gradient(cmap="YlGnBu", axis=1),
                    use_container_width=True
                )

            # B) Weekly Cumulative Growth from D0
            st.markdown("---"); st.markdown("### B) Weekly Cumulative Growth from D0")
            for wk in sorted(weeks_selected):
                sub = wk_df[(wk_df[wcol]==wk) & (wk_df["campaign_network"].isin(selected_campaigns))]
                if sub.empty:
                    st.info(f"No data for week {wk}")
                    continue
                d0_key = next((nm for nm in roas_columns_filtered if str(nm).lower()=="roas_d0"), None)
                if d0_key is None:
                    st.info("D0 not found."); break
                gr = sub.copy()
                for c in roas_columns_filtered:
                    if str(c).lower()=="roas_d0": continue
                    gr[c] = np.where(gr[d0_key]>0, gr[c]/gr[d0_key]-1, np.nan)
                growth_cols = [c for c in roas_columns_filtered if str(c).lower()!="roas_d0"]
                tblg = gr.groupby("campaign_network")[growth_cols].mean(numeric_only=True)
                st.markdown(f"**Week: {wk}**")
                st.dataframe(
                    tblg.style.format("{:.2f}").background_gradient(cmap="YlGnBu", axis=1),
                    use_container_width=True
                )
        else:
            st.info("Select one or more campaigns to view weekly breakdown.")
