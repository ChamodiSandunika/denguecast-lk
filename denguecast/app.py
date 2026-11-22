"""
DengueCast LK — Ensemble Learning Dashboard (Streamlit)
Run:  streamlit run app.py
Front-end code generated with the help of Claude (Anthropic) — see README "AI Usage Acknowledgment".
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dengue_features import FEATURES, build_features, make_row, risk_level

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b"
RISK_COL = {"Low": "#1baf7a", "Moderate": "#eda100", "High": "#eb6834", "Very high": "#d03b3b"}
RISK_ICON = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Very high": "🔴"}

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Bagging (Decision Trees)": "bagging.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "CatBoost": "catboost.pkl",
    "Stacking (RF+XGB+LGBM+Cat+Ridge)": "stacking.pkl",
    "Hybrid B: RF + ANN": "rf_ann_hybrid.pkl",
    "ANN (MLP)": "ann.pkl",
    "Ridge regression": "ridge.pkl",
}
FAMILY = {"Random Forest": "Bagging", "Bagging (Decision Trees)": "Bagging", "XGBoost": "Boosting",
          "LightGBM": "Boosting", "CatBoost": "Boosting", "Stacking (RF+XGB+LGBM+Cat+Ridge)": "Stacking",
          "Hybrid A: ARIMA + RF": "Hybrid", "Hybrid B: RF + ANN": "Hybrid", "ANN (MLP)": "Neural net",
          "Ridge regression": "Baseline", "Naive (last week)": "Baseline", "ARIMA (per division)": "Time series"}

st.set_page_config(page_title="DengueCast LK", page_icon="🦟", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_data
def load_data():
    df = pd.read_csv(P("data", "dengue_weekly_clean.csv"), parse_dates=["start_date", "end_date"])
    res = pd.read_csv(P("results", "model_comparison.csv"))
    per = pd.read_csv(P("results", "error_by_period.csv"), index_col=0)
    preds = pd.read_csv(P("results", "test_predictions.csv"), parse_dates=["start_date"])
    thr = pd.read_csv(P("results", "division_thresholds.csv"), index_col=0)
    fc = pd.read_csv(P("results", "next_week_forecast.csv"), parse_dates=["start_date"])
    imp = pd.read_csv(P("results", "feature_importance.csv"), index_col=0)
    perm = pd.read_csv(P("results", "permutation_importance.csv"), index_col=0).iloc[:, 0]
    info = json.load(open(P("results", "run_info.json")))
    return df, res, per, preds, thr, fc, imp, perm, info


@st.cache_resource
def load_models():
    models = {}
    for name, f in MODEL_FILES.items():
        path = P("models", f)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


df, res, per, preds, thr, fc, imp, perm, info = load_data()
models = load_models()
best = info["best_model"]


def style_fig(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      font=dict(size=13), legend=dict(orientation="h", y=1.08, x=0),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#ececec")
    return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🦟 DengueCast LK")
st.sidebar.caption("Ensemble learning early warning for weekly dengue cases in Sri Lanka's 26 RDHS divisions")
page = st.sidebar.radio("Page", ["Overview", "Model comparison", "Forecast explorer", "Feature importance",
                                 "Next-week early warning", "Live prediction", "About"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best model (MAE):** {best}")
st.sidebar.markdown(f"**Data:** {df.start_date.min():%d %b %Y} – {df.start_date.max():%d %b %Y}")
st.sidebar.caption("Source: Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health Sri Lanka "
                   "(via the denguedatahub package).")

# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("Dengue in Sri Lanka — dataset overview")
    nat = df.groupby("start_date")["cases"].sum()
    y26 = df[df.year == 2026]["cases"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (division-weeks)", f"{len(df):,}")
    c2.metric("Divisions", df.district.nunique())
    c3.metric("Weeks", f"{df.start_date.nunique():,}")
    c4.metric("Cases in 2026 (to mid-Aug)", f"{y26:,}")

    fig = go.Figure(go.Scatter(x=nat.index, y=nat.values, line=dict(color=BLUE, width=2),
                               hovertemplate="%{x|%d %b %Y}<br>%{y:,} cases<extra></extra>"))
    fig.update_layout(title="Weekly dengue cases — whole island")
    st.plotly_chart(style_fig(fig), width="stretch")

    a, b = st.columns(2)
    yearly = df.groupby("year")["cases"].sum().reset_index()
    fig = px.bar(yearly, x="year", y="cases", title="Cases per year", color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="%{x}: %{y:,} cases<extra></extra>")
    a.plotly_chart(style_fig(fig, 340), width="stretch")

    yr = b.selectbox("Year for division totals", sorted(df.year.unique(), reverse=True))
    dv = df[df.year == yr].groupby("district")["cases"].sum().sort_values().reset_index()
    fig = px.bar(dv, x="cases", y="district", orientation="h", title=f"Cases by division, {yr}",
                 color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="%{y}: %{x:,}<extra></extra>")
    b.plotly_chart(style_fig(fig, 560), width="stretch")

    st.subheader("Seasonality")
    s = df[df.year.between(2010, 2025)].copy()
    s["month"] = s.start_date.dt.month
    m = s.groupby(["year", "month"])["cases"].sum().unstack()
    m = (m.div(m.sum(axis=1), axis=0) * 100).round(1)
    fig = px.imshow(m, color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Month", y="Year", color="% of year"),
                    title="Share of each year's cases by month — peaks after the SW (Jun–Jul) and NE (Dec–Jan) monsoons")
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    with st.expander("Sample rows"):
        st.dataframe(df.tail(30), width="stretch")

elif page == "Model comparison":
    st.title("Model comparison — test period " + " to ".join(info["test_period"]))
    st.caption("1-week-ahead forecast for every division. Lower MAE/RMSE is better. "
               "Alert recall = share of outbreak weeks (above the division's 90th percentile) that the forecast also flagged.")
    show = res.drop(columns=["rank"]).copy()
    show.insert(0, "★", np.where(show.model == best, "★ best", ""))
    st.dataframe(show.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}", "WAPE": "{:.1f}%",
                                    "Alert_Recall": "{:.3f}", "Alert_Precision": "{:.3f}", "Alert_F1": "{:.3f}",
                                    "Train_time_s": "{:.1f}"})
                 .highlight_min(subset=["MAE", "RMSE", "WAPE"], color="#fde3d6")
                 .highlight_max(subset=["R2", "Alert_Recall", "Alert_F1"], color="#fde3d6"),
                 width="stretch", hide_index=True)

    metric = st.radio("Metric", ["MAE", "RMSE", "R2", "WAPE", "Alert_Recall", "Alert_F1", "Train_time_s"],
                      horizontal=True)
    asc = metric in ("MAE", "RMSE", "WAPE", "Train_time_s")
    d = res.sort_values(metric, ascending=not asc)
    colors = [ORANGE if m == best else (GREY if FAMILY.get(m) == "Baseline" else BLUE) for m in d.model]
    fig = go.Figure(go.Bar(x=d[metric], y=d.model, orientation="h", marker_color=colors,
                           text=d[metric].round(3), textposition="outside",
                           hovertemplate="%{y}<br>" + metric + ": %{x:.3f}<extra></extra>"))
    fig.update_layout(title=f"{metric} by model (orange = best by MAE, grey = baselines)")
    st.plotly_chart(style_fig(fig, 480), width="stretch")

    st.subheader("Error by period: normal years vs the 2026 outbreak (MAE)")
    pl = per.reset_index().melt(id_vars=per.index.name or "index", var_name="period", value_name="MAE")
    pl.columns = ["model", "period", "MAE"]
    fig = px.bar(pl, x="MAE", y="model", color="period", barmode="group", orientation="h",
                 color_discrete_sequence=[BLUE, AQUA, ORANGE])
    st.plotly_chart(style_fig(fig, 620), width="stretch")
    st.info("All models are accurate in 2024–2025 (MAE ≈ 8–10) but the error rises to 34–41 in the 2026 outbreak. "
            "Stacking and LightGBM handle the outbreak best; ARIMA is worst because it is linear and sees one division at a time.")

elif page == "Forecast explorer":
    st.title("Forecast explorer — actual vs predicted")
    model_cols = [c for c in preds.columns if c not in ("start_date", "district", "epi_week", "cases")]
    a, b = st.columns([1, 2])
    dist = a.selectbox("Division", sorted(preds.district.unique()), index=sorted(preds.district.unique()).index("Colombo"))
    chosen = b.multiselect("Models", model_cols, default=[best, "Random Forest"])
    d = preds[preds.district == dist]
    fig = go.Figure(go.Scatter(x=d.start_date, y=d.cases, name="Actual", line=dict(color=INK, width=2.5)))
    palette = [ORANGE, BLUE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for i, m in enumerate(chosen):
        fig.add_trace(go.Scatter(x=d.start_date, y=d[m], name=m, line=dict(color=palette[i % 8], width=2)))
    fig.update_layout(title=f"{dist}: weekly cases, test period", hovermode="x unified")
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    if chosen:
        rows = [{"model": m, "MAE (this division)": np.abs(d.cases - d[m]).mean(),
                 "Max miss": np.abs(d.cases - d[m]).max()} for m in chosen]
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    fig = px.scatter(d, x="cases", y=chosen[0] if chosen else best, opacity=0.7,
                     color_discrete_sequence=[BLUE], title="Predicted vs actual (closer to the diagonal is better)")
    mx = float(max(d.cases.max(), 1))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=GREY, dash="dash"), showlegend=False))
    st.plotly_chart(style_fig(fig, 420), width="stretch")

elif page == "Feature importance":
    st.title("Which inputs drive the forecast?")
    a, b = st.columns(2)
    for col, c in zip((a, b), imp.columns):
        s = imp[c].sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=BLUE))
        fig.update_layout(title=f"Built-in importance — {c}")
        col.plotly_chart(style_fig(fig, 460), width="stretch")
    s = perm.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=ORANGE))
    fig.update_layout(title=f"Permutation importance — {best} (extra MAE when a feature is shuffled)")
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    st.markdown("- **lag1 / roll_mean4**: the last weeks' cases — dengue transmission runs in ~2-week cycles.\n"
                "- **district**: very different baselines (dense Colombo vs rural Mannar).\n"
                "- **nat_lag1**: island-wide outbreaks lift every division.\n"
                "- Calendar/season adds little because its effect is already inside recent counts.")

elif page == "Next-week early warning":
    wk = fc.start_date.iloc[0]
    st.title(f"Early warning — week starting {wk:%d %b %Y}")
    st.caption(f"Forecast from the last complete week in the data, using {best}. "
               "Risk = forecast compared with the division's own history (2008–2023): "
               "Low < median ≤ Moderate < 75th pct ≤ High < 95th pct ≤ Very high.")
    counts = fc.risk.value_counts()
    cols = st.columns(4)
    for c, r in zip(cols, ["Very high", "High", "Moderate", "Low"]):
        c.metric(f"{RISK_ICON[r]} {r}", int(counts.get(r, 0)))
    d = fc.sort_values("forecast")
    fig = go.Figure(go.Bar(x=d.forecast, y=d.district, orientation="h",
                           marker_color=[RISK_COL[r] for r in d.risk], text=d.risk, textposition="outside",
                           customdata=d.last_week_cases,
                           hovertemplate="%{y}<br>Forecast: %{x:.0f}<br>Last week: %{customdata}<extra></extra>"))
    fig.update_layout(title="Forecast cases next week (colour = risk level)")
    st.plotly_chart(style_fig(fig, 720), width="stretch")
    tab = fc.copy()
    tab["risk"] = tab.risk.map(lambda r: f"{RISK_ICON[r]} {r}")
    st.dataframe(tab.drop(columns=["start_date"]).round(1), hide_index=True, width="stretch")
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "next_week_forecast.csv")

elif page == "Live prediction":
    st.title("Live prediction")
    st.caption("Enter the latest weekly counts (e.g. from the newest Weekly Epidemiological Report). "
               "Values are pre-filled from the data for the week you pick.")
    dists = sorted(df.district.unique())
    a, b = st.columns(2)
    dist = a.selectbox("Division", dists, index=dists.index("Colombo"))
    weeks = sorted(df.start_date.unique())[60:]
    base = b.select_slider("Forecast the week after", options=weeks, value=weeks[-1],
                           format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    base = pd.Timestamp(base)
    h = df[df.district == dist].set_index("start_date")["cases"]
    hist = h.loc[:base]
    nat = df.groupby("start_date")["cases"].sum()
    target_date = base + pd.Timedelta(days=7)
    ep = int(df.loc[df.start_date == base, "epi_week"].iloc[0]) % 52 + 1
    actual_next = h.get(target_date, np.nan)

    st.markdown(f"**Forecasting week starting {target_date:%d %b %Y}** (epi week {ep}) for **{dist}**")
    c = st.columns(4)
    last4 = [c[i].number_input(f"Cases {i + 1} week{'s' if i else ''} ago", min_value=0,
                               value=int(hist.iloc[-1 - i]), step=1, key=f"l{i}") for i in range(4)]
    c = st.columns(3)
    w58 = c[0].number_input("Average cases 5–8 weeks ago", min_value=0.0, value=float(hist.iloc[-8:-4].mean()))
    ly = c[1].number_input("Cases same week last year", min_value=0, value=int(hist.iloc[-52]) if len(hist) > 52 else 0)
    nl = c[2].number_input("National total last week", min_value=0, value=int(nat.loc[base]))

    X = make_row(dist, ep, target_date.month, last4, ly, w58, nl)
    out = []
    for name, m in models.items():
        out.append({"model": name, "family": FAMILY[name], "forecast": float(np.clip(m.predict(X)[0], 0, None))})
    out = pd.DataFrame(out).sort_values("forecast", ascending=False)
    main = out.loc[out.model == best, "forecast"].iloc[0] if best in out.model.values else out.forecast.median()
    risk = risk_level(main, thr.loc[dist])
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Forecast ({best.split(' (')[0]})", f"{main:.0f} cases", f"{main - last4[0]:+.0f} vs last week")
    k2.metric("Risk level", f"{RISK_ICON[risk]} {risk}")
    k3.metric("Actual (if known)", "—" if np.isnan(actual_next) else f"{actual_next:.0f} cases")
    colors = [ORANGE if m == best else BLUE for m in out.model]
    fig = go.Figure(go.Bar(x=out.forecast, y=out.model, orientation="h", marker_color=colors,
                           text=out.forecast.round(1), textposition="outside"))
    fig.update_layout(title="Forecast from every saved model (.pkl)")
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.caption(f"Division thresholds — median {thr.loc[dist, 'p50']:.0f}, 75th pct {thr.loc[dist, 'p75']:.0f}, "
               f"95th pct {thr.loc[dist, 'p95']:.0f} cases/week. "
               "Hybrid A (ARIMA + RF) needs the full series and is shown on the comparison and early-warning pages only.")

else:
    st.title("About this project")
    st.markdown(f"""
**DengueCast LK** compares bagging, boosting, stacking and hybrid ensembles for 1-week-ahead dengue forecasting in all
26 RDHS divisions of Sri Lanka.

- **Bagging:** Random Forest, Bagged Decision Trees
- **Boosting:** XGBoost, LightGBM, CatBoost
- **Stacking:** RF + XGBoost + LightGBM + CatBoost + Ridge → non-negative linear meta-learner
  (weights: {", ".join(f"{k} {v:.2f}" for k, v in info["stack_weights"].items())})
- **Hybrids:** ARIMA + RF (RF learns ARIMA residuals), RF + ANN (ANN uses RF out-of-bag predictions)
- **Train:** {" to ".join(info["train_period"])} · **Test:** {" to ".join(info["test_period"])}

**Data:** Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health Sri Lanka, via the
[denguedatahub](https://github.com/thiyangt/denguedatahub) R package. The last four provisional weeks were removed.

**AI usage:** code for this dashboard and the modelling notebook was written with help from Claude (Anthropic).
See the README for the full acknowledgment.
""")
