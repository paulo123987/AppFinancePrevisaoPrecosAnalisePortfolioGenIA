"""
Unified Finance App – Previsão de Preços + Análise de Portfólio + Relatório IA
Dados: Yahoo Finance (yfinance) – 100% gratuito
Tema executivo: branco/cinza claro, texto preto/vermelho
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & EXECUTIVE THEME
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Intelligence Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXECUTIVE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1A1A1A;
}
.stApp {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7F7F8 100%);
}

/* ── Sidebar ───────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #F0F0F2;
    border-right: 1px solid #E0E0E0;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1A1A1A;
}

/* ── Headers ───────────────────────────────────── */
h1, h2, h3 { color: #1A1A1A !important; font-weight: 700 !important; }
h1 { font-size: 2rem !important; letter-spacing: -0.5px; }

/* ── Red accents ───────────────────────────────── */
.red-accent { color: #D32F2F; font-weight: 700; }
.red-badge {
    background: #D32F2F; color: #FFF; padding: 4px 14px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    display: inline-block; margin-bottom: 8px;
}
.red-divider { border-top: 3px solid #D32F2F; margin: 30px 0; }

/* ── Cards ─────────────────────────────────────── */
.exec-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E8;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.exec-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.exec-card h4 { color: #1A1A1A; margin-bottom: 8px; }
.exec-card .metric-value { font-size: 2rem; font-weight: 800; color: #D32F2F; }
.exec-card .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }

/* ── Metric override ───────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E8E8;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.8px; }
[data-testid="stMetricValue"] { color: #1A1A1A !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Buttons ───────────────────────────────────── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.8rem;
    transition: all 0.2s;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #D32F2F !important;
    color: #FFF !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #B71C1C !important;
    box-shadow: 0 4px 12px rgba(211,47,47,0.3);
}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: #F5F5F5 !important;
    color: #1A1A1A !important;
    border: 1px solid #DDD !important;
}

/* ── Section divider ───────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 40px 0 20px 0;
}
.section-header .icon {
    background: #D32F2F; color: #FFF;
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.section-header .title { font-size: 1.4rem; font-weight: 700; color: #1A1A1A; }
.section-header .subtitle { font-size: 0.85rem; color: #999; }

/* ── Expander ──────────────────────────────────── */
.streamlit-expanderHeader { font-weight: 600 !important; color: #1A1A1A !important; }

/* ── Progress / spinner ────────────────────────── */
.stSpinner > div > div { border-top-color: #D32F2F !important; }

/* ── Plotly override ───────────────────────────── */
.js-plotly-plot .plotly .modebar { top: 8px !important; right: 8px !important; }
</style>
"""
st.markdown(EXECUTIVE_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
POPULAR_TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "WEGE3.SA", "MGLU3.SA", "B3SA3.SA", "RENT3.SA", "SUZB3.SA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
    "JPM", "V", "JNJ", "SPY", "QQQ","AZTE3.SA",
]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", color="#1A1A1A"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAFA",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#FFF", font_size=12, font_family="Inter"),
)

RED = "#D32F2F"
DARK = "#1A1A1A"
GREY = "#888888"
COLORS = ["#D32F2F", "#1A1A1A", "#6D6D6D", "#B0B0B0", "#E57373"]


def section_header(icon: str, title: str, subtitle: str = ""):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""<div class="section-header">
            <div class="icon">{icon}</div>
            <div><div class="title">{title}</div>{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str):
    st.markdown(
        f"""<div class="exec-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def download_data(tickers: list, start: str, end: str, interval: str) -> dict:
    """Download data from yfinance for each ticker; returns dict of DataFrames."""
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, interval=interval, progress=False)
            if df is not None and not df.empty:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            st.warning(f"Erro ao baixar {t}: {e}")
    return data


def build_lstm_model(input_shape):
    """Build a simple LSTM model for price prediction."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_predict_ticker(df: pd.DataFrame, ticker: str, look_back: int = 60, epochs: int = 10):
    """Train LSTM on a single ticker and return results dict."""
    close = df[["Close"]].dropna().values
    if len(close) < look_back + 20:
        return None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    train_size = int(len(scaled) * 0.8)
    train_data = scaled[:train_size]
    test_data = scaled[train_size - look_back:]

    X_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        X_train.append(train_data[i - look_back:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_test, y_test_raw = [], []
    for i in range(look_back, len(test_data)):
        X_test.append(test_data[i - look_back:i, 0])
        y_test_raw.append(test_data[i, 0])
    X_test, y_test_raw = np.array(X_test), np.array(y_test_raw)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)

    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    actual = scaler.inverse_transform(y_test_raw.reshape(-1, 1)).flatten()

    mse = float(np.mean((actual - predictions) ** 2))
    mape = float(np.mean(np.abs((actual - predictions) / (actual + 1e-9))) * 100)
    accuracy = round(100 - mape, 2)

    test_dates = df.index[train_size:]
    if len(test_dates) > len(actual):
        test_dates = test_dates[: len(actual)]
    elif len(test_dates) < len(actual):
        actual = actual[: len(test_dates)]
        predictions = predictions[: len(test_dates)]

    return {
        "ticker": ticker,
        "actual": actual,
        "predicted": predictions,
        "dates": test_dates,
        "mse": mse,
        "mape": mape,
        "accuracy": accuracy,
        "train_dates": df.index[:train_size],
        "train_close": df["Close"].values[:train_size],
        "last_price": float(df["Close"].iloc[-1]),
        "first_price": float(df["Close"].iloc[0]),
    }


# ──────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────
for key in [
    "module1_results", "module1_done", "module2_done", "module2_results",
    "module3_done", "module3_report", "selected_tickers", "start_date",
    "end_date", "granularity", "proceed_to_m2", "data_cache",
]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ("module1_done", "module2_done", "module3_done", "proceed_to_m2") else False

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown('<span class="red-badge">EXECUTIVE DASHBOARD</span>', unsafe_allow_html=True)
st.markdown("# 📊 Finance Intelligence Suite")
st.caption("Previsão de preços · Análise de portfólio · Relatório IA — powered by Yahoo Finance")
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SIDEBAR – PARAMETERS
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parâmetros")
    st.markdown("---")

    tickers = st.multiselect(
        "🔎 Tickers (máx. 5)",
        options=POPULAR_TICKERS,
        default=["AAPL", "MSFT","AZTE3.SA","PETR4.SA", "VALE3.SA"],
        max_selections=5,
        help="Selecione até 5 tickers disponíveis no Yahoo Finance",
    )
    custom_ticker = st.text_input("Ou digite um ticker personalizado", placeholder="ex: AMZN")
    if custom_ticker:
        custom_ticker = custom_ticker.strip().upper()
        if custom_ticker not in tickers and len(tickers) < 5:
            tickers.append(custom_ticker)

    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("📅 Início", value=datetime.now() - timedelta(days=59))
    with col_d2:
        end_date = st.date_input("📅 Fim", value=datetime.now())

    granularity = st.radio(
        "📐 Granularidade",
        ["Diário (1d)", "Intradiário 30min (30m)"],
        index=0,
    )
    interval = "1d" if "1d" in granularity else "30m"

    if interval == "30m":
        max_intra = datetime.now() - timedelta(days=59)
        if start_date < max_intra.date():
            st.warning("⚠️ Dados intradiários de 30min estão disponíveis apenas para os últimos ~60 dias. A data de início será ajustada automaticamente.")
            start_date = max_intra.date()

    st.markdown("---")
    run_prediction = st.button("▶️ Executar Previsão", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**Desenvolvido por**")
    st.caption("Equipe de IA Financeira · v2.0")
    st.caption(f"Dados: Yahoo Finance (yfinance)")

# ──────────────────────────────────────────────────────────────
# MODULE 1 – STOCK PRICE PREDICTION (LSTM)
# ──────────────────────────────────────────────────────────────
if run_prediction and tickers:
    st.session_state.selected_tickers = tickers
    st.session_state.start_date = str(start_date)
    st.session_state.end_date = str(end_date)
    st.session_state.granularity = interval
    st.session_state.module1_done = False
    st.session_state.proceed_to_m2 = False
    st.session_state.module2_done = False
    st.session_state.module3_done = False
    st.session_state.module2_results = None
    st.session_state.module3_report = None

    section_header("📊", "Módulo 1 — Previsão de Preços", "LSTM neural network · yfinance")

    with st.spinner("Baixando dados do Yahoo Finance..."):
        data = download_data(tickers, str(start_date), str(end_date), interval)
        st.session_state.data_cache = data

    if not data:
        st.error("Nenhum dado encontrado para os tickers selecionados. Verifique os tickers e o período.")
    else:
        results = []
        progress = st.progress(0, text="Treinando modelos LSTM...")
        for idx, (ticker, df) in enumerate(data.items()):
            progress.progress((idx + 1) / len(data), text=f"Treinando {ticker}...")
            res = train_predict_ticker(df, ticker)
            if res:
                results.append(res)
        progress.empty()

        if results:
            st.session_state.module1_results = results
            st.session_state.module1_done = True
            st.success(f"✅ Previsão concluída para {len(results)} ticker(s)!")
        else:
            st.error("Não foi possível treinar modelos. Dados insuficientes para o período selecionado.")

# ── DISPLAY MODULE 1 RESULTS ──
if st.session_state.get("module1_done") and st.session_state.get("module1_results"):
    section_header("📊", "Módulo 1 — Resultados da Previsão", "LSTM · Valores reais vs previstos")

    results = st.session_state.module1_results

    # Overview metrics
    mcols = st.columns(len(results))
    for i, res in enumerate(results):
        with mcols[i]:
            delta = res["last_price"] - res["first_price"]
            delta_pct = (delta / res["first_price"]) * 100 if res["first_price"] != 0 else 0
            st.metric(
                label=res["ticker"],
                value=f"${res['last_price']:.2f}",
                delta=f"{delta_pct:+.1f}%",
            )

    # Charts per ticker
    for res in results:
        with st.expander(f"📈 {res['ticker']} — Detalhes", expanded=True):
            fig = go.Figure()
            # Training data
            fig.add_trace(go.Scatter(
                x=res["train_dates"], y=res["train_close"],
                name="Treino", line=dict(color=GREY, width=1.5),
                opacity=0.5,
            ))
            # Actual test
            fig.add_trace(go.Scatter(
                x=res["dates"], y=res["actual"],
                name="Real (Teste)", line=dict(color=DARK, width=2.5),
            ))
            # Predicted
            fig.add_trace(go.Scatter(
                x=res["dates"], y=res["predicted"],
                name="Previsão LSTM", line=dict(color=RED, width=2.5, dash="dash"),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"{res['ticker']} — Preço Real vs Previsão", font=dict(size=16)),
                xaxis_title="Data", yaxis_title="Preço (USD)",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("MSE", f"{res['mse']:.4f}")
            m2.metric("MAPE", f"{res['mape']:.2f}%")
            m3.metric("Acurácia", f"{res['accuracy']:.2f}%")

    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

    # Flow control buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ Continuar para Análise de Portfólio", type="primary", use_container_width=True):
            st.session_state.proceed_to_m2 = True
            st.rerun()
    with col_btn2:
        if st.button("🔁 Refazer Previsão", type="secondary", use_container_width=True):
            st.session_state.module1_done = False
            st.session_state.module1_results = None
            st.session_state.proceed_to_m2 = False
            st.session_state.module2_done = False
            st.session_state.module2_results = None
            st.session_state.module3_done = False
            st.session_state.module3_report = None
            st.rerun()

# ──────────────────────────────────────────────────────────────
# MODULE 2 – PORTFOLIO ANALYSIS
# ──────────────────────────────────────────────────────────────
if st.session_state.get("proceed_to_m2") and st.session_state.get("module1_done"):

    if not st.session_state.get("module2_done"):
        section_header("💼", "Módulo 2 — Análise de Portfólio", "Regressão Linear · Markowitz · Métricas de Risco")

        tickers_m2 = st.session_state.selected_tickers
        data = st.session_state.data_cache

        with st.spinner("Calculando métricas de portfólio..."):
            # ── Build returns DataFrame ──
            close_df = pd.DataFrame()
            for t in tickers_m2:
                if t in data and "Close" in data[t].columns:
                    close_df[t] = data[t]["Close"]
            close_df = close_df.dropna()

            if close_df.empty or len(close_df) < 10:
                st.error("Dados insuficientes para análise de portfólio.")
            else:
                returns = close_df.pct_change().dropna()

                # ── Download benchmark SPY ──
                spy_data = download_data(
                    ["SPY"],
                    st.session_state.start_date,
                    st.session_state.end_date,
                    st.session_state.granularity,
                )
                has_spy = "SPY" in spy_data and not spy_data["SPY"].empty
                if has_spy:
                    spy_close = spy_data["SPY"]["Close"].reindex(returns.index).ffill().bfill()
                    spy_returns = spy_close.pct_change().dropna()
                    common_idx = returns.index.intersection(spy_returns.index)
                    returns = returns.loc[common_idx]
                    spy_returns = spy_returns.loc[common_idx]

                # ── Correlation Heatmap ──
                st.markdown("### 🔗 Matriz de Correlação")
                corr = returns.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale=["#FFFFFF", "#E57373", "#D32F2F"],
                    aspect="auto",
                )
                fig_corr.update_layout(**PLOTLY_LAYOUT, height=400, title="Correlação entre Retornos")
                st.plotly_chart(fig_corr, use_container_width=True)

                # ── Risk Metrics ──
                st.markdown("### 📏 Métricas de Risco")
                n_tickers = len(tickers_m2)
                equal_weights = np.array([1.0 / n_tickers] * n_tickers)
                port_returns = returns.dot(equal_weights)

                annual_factor = 252 if st.session_state.granularity == "1d" else 252 * 13  # 13 slots of 30min per day
                annual_return = float(port_returns.mean() * annual_factor)
                annual_vol = float(port_returns.std() * np.sqrt(annual_factor))
                sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

                cumulative = (1 + port_returns).cumprod()
                roll_max = cumulative.cummax()
                drawdown = (cumulative - roll_max) / roll_max
                max_drawdown = float(drawdown.min())

                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("Retorno Anual", f"{annual_return:.2%}")
                rm2.metric("Volatilidade", f"{annual_vol:.2%}")
                rm3.metric("Sharpe Ratio", f"{sharpe:.3f}")
                rm4.metric("Max Drawdown", f"{max_drawdown:.2%}")

                # ── Beta & Alpha per ticker ──
                if has_spy and len(spy_returns) > 0:
                    st.markdown("### 📊 Beta & Alpha por Ticker")
                    beta_alpha = []
                    for t in tickers_m2:
                        if t in returns.columns:
                            cov = np.cov(returns[t].values, spy_returns.values)
                            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
                            alpha = float(returns[t].mean() * annual_factor - beta * spy_returns.mean() * annual_factor)
                            beta_alpha.append({"Ticker": t, "Beta": round(beta, 4), "Alpha (anual)": f"{alpha:.4f}"})
                    if beta_alpha:
                        st.dataframe(pd.DataFrame(beta_alpha), use_container_width=True, hide_index=True)

                # ── Linear Regression: Portfolio vs Market ──
                if has_spy and len(spy_returns) >= 10:
                    st.markdown("### 📈 Regressão Linear — Portfólio vs Mercado (SPY)")
                    X_reg = spy_returns.values.reshape(-1, 1)
                    y_reg = port_returns.values

                    split = int(len(X_reg) * 0.7)
                    X_tr, X_te = X_reg[:split], X_reg[split:]
                    y_tr, y_te = y_reg[:split], y_reg[split:]

                    reg = LinearRegression().fit(X_tr, y_tr)
                    y_pred = reg.predict(X_te)

                    mse_reg = mean_squared_error(y_te, y_pred)
                    r2_reg = r2_score(y_te, y_pred)

                    fig_reg = go.Figure()
                    fig_reg.add_trace(go.Scatter(
                        y=y_te, mode="lines+markers", name="Retorno Real",
                        marker=dict(size=4, color=DARK), line=dict(width=2, color=DARK),
                    ))
                    fig_reg.add_trace(go.Scatter(
                        y=y_pred, mode="lines+markers", name="Retorno Predito",
                        marker=dict(size=4, color=RED), line=dict(width=2, color=RED, dash="dash"),
                    ))
                    fig_reg.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Retorno Real vs Predito (Regressão Linear)",
                        xaxis_title="Amostra (Test Set)",
                        yaxis_title="Retorno Diário",
                        height=400,
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)

                    rc1, rc2 = st.columns(2)
                    rc1.metric("MSE (Regressão)", f"{mse_reg:.6f}")
                    rc2.metric("R² Score", f"{r2_reg:.4f}")

                # ── Efficient Frontier (Markowitz) ──
                if n_tickers >= 2:
                    st.markdown("### 🎯 Fronteira Eficiente de Markowitz")
                    mean_returns = returns.mean() * annual_factor
                    cov_matrix = returns.cov() * annual_factor

                    def portfolio_stats(weights):
                        ret = np.dot(weights, mean_returns)
                        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        return ret, vol

                    def neg_sharpe(weights):
                        ret, vol = portfolio_stats(weights)
                        return -(ret - 0.02) / vol if vol > 0 else 0

                    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
                    bounds = tuple((0, 1) for _ in range(n_tickers))
                    init = equal_weights

                    # Optimal portfolio
                    opt = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)
                    opt_weights = opt.x
                    opt_ret, opt_vol = portfolio_stats(opt_weights)

                    # Random portfolios for frontier
                    n_sim = 3000
                    sim_rets, sim_vols, sim_sharpes = [], [], []
                    for _ in range(n_sim):
                        w = np.random.random(n_tickers)
                        w /= w.sum()
                        r, v = portfolio_stats(w)
                        sim_rets.append(r)
                        sim_vols.append(v)
                        sim_sharpes.append((r - 0.02) / v if v > 0 else 0)

                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=sim_vols, y=sim_rets, mode="markers",
                        marker=dict(size=3, color=sim_sharpes, colorscale=[[0, "#E0E0E0"], [0.5, "#E57373"], [1, "#D32F2F"]], showscale=True, colorbar=dict(title="Sharpe")),
                        name="Portfólios Simulados",
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[opt_vol], y=[opt_ret], mode="markers",
                        marker=dict(size=16, color=RED, symbol="star", line=dict(width=2, color=DARK)),
                        name="Portfólio Ótimo",
                    ))
                    fig_ef.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Fronteira Eficiente",
                        xaxis_title="Volatilidade (Risco)",
                        yaxis_title="Retorno Esperado",
                        height=500,
                    )
                    st.plotly_chart(fig_ef, use_container_width=True)

                    # Optimal weights
                    st.markdown("#### 🏆 Pesos Ótimos do Portfólio")
                    w_df = pd.DataFrame({
                        "Ticker": tickers_m2,
                        "Peso (%)": [f"{w * 100:.1f}%" for w in opt_weights],
                    })
                    st.dataframe(w_df, use_container_width=True, hide_index=True)

                    om1, om2, om3 = st.columns(3)
                    om1.metric("Retorno Esperado", f"{opt_ret:.2%}")
                    om2.metric("Volatilidade", f"{opt_vol:.2%}")
                    om3.metric("Sharpe Ratio", f"{(opt_ret - 0.02) / opt_vol:.3f}" if opt_vol > 0 else "N/A")

                # ── Store Module 2 results ──
                m2_results = {
                    "tickers": tickers_m2,
                    "annual_return": annual_return,
                    "annual_vol": annual_vol,
                    "sharpe": sharpe,
                    "max_drawdown": max_drawdown,
                    "correlation": corr.to_dict(),
                }
                if n_tickers >= 2:
                    m2_results["optimal_weights"] = {t: float(w) for t, w in zip(tickers_m2, opt_weights)}
                    m2_results["opt_return"] = float(opt_ret)
                    m2_results["opt_vol"] = float(opt_vol)
                if has_spy and len(spy_returns) >= 10:
                    m2_results["regression_mse"] = float(mse_reg)
                    m2_results["regression_r2"] = float(r2_reg)
                if has_spy and beta_alpha:
                    m2_results["beta_alpha"] = beta_alpha

                st.session_state.module2_results = m2_results
                st.session_state.module2_done = True
                st.success("✅ Análise de portfólio concluída!")

    # Show Module 2 results area (already shown above inside the if block)

# ──────────────────────────────────────────────────────────────
# MODULE 3 – AI REPORT (Multi-Agent)
# ──────────────────────────────────────────────────────────────
if st.session_state.get("module2_done"):
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)
    section_header("🤖", "Módulo 3 — Relatório Executivo com IA", "Multi-agentes · OpenAI / Groq")

    # API Selection
    api_choice = st.radio(
        "🔑 Escolha o provedor de IA",
        ["OpenAI (gpt-4o-mini)", "Groq (llama-3.3-70b-versatile)"],
        horizontal=True,
    )

    if st.button("🤖 Gerar Relatório Executivo", type="primary", use_container_width=True):
        
        # --- ALTERAÇÃO FEITA AQUI ---
        # Tenta carregar dados do arquivo .env (se estiver rodando no computador local)
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        except ImportError:
            # Se a biblioteca dotenv não existir (como no Streamlit Cloud sem ela no requirements.txt), ignora.
            pass

        # Usa st.secrets.get() para buscar primeiro nas Secrets do Streamlit Cloud
        # Caso não ache, o fallback (plano B) é o os.getenv() para buscar do .env local
        if "OpenAI" in api_choice:
            api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            if not api_key:
                st.error("🔑 OPENAI_API_KEY não encontrada nas secrets do Streamlit nem no arquivo .env.")
                st.stop()
        else:
            api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
            if not api_key:
                st.error("🔑 GROQ_API_KEY não encontrada nas secrets do Streamlit nem no arquivo .env.")
                st.stop()
        # --- FIM DA ALTERAÇÃO ---

        # ── Agent 1: Collector ──
        with st.spinner("🔍 Agent 1 — Coletando dados dos módulos..."):
            m1 = st.session_state.module1_results
            m2 = st.session_state.module2_results

            context_parts = ["## Dados Coletados para Análise\n"]
            context_parts.append("### Módulo 1 — Previsão de Preços (LSTM)")
            for r in m1:
                context_parts.append(
                    f"- **{r['ticker']}**: Último preço ${r['last_price']:.2f}, "
                    f"Primeiro preço ${r['first_price']:.2f}, "
                    f"Variação {((r['last_price'] - r['first_price']) / r['first_price'] * 100):.1f}%, "
                    f"MSE={r['mse']:.4f}, MAPE={r['mape']:.2f}%, Acurácia={r['accuracy']:.2f}%"
                )

            context_parts.append("\n### Módulo 2 — Análise de Portfólio")
            context_parts.append(f"- Retorno Anual: {m2['annual_return']:.2%}")
            context_parts.append(f"- Volatilidade: {m2['annual_vol']:.2%}")
            context_parts.append(f"- Sharpe Ratio: {m2['sharpe']:.3f}")
            context_parts.append(f"- Max Drawdown: {m2['max_drawdown']:.2%}")

            if "optimal_weights" in m2:
                context_parts.append(f"- Retorno Ótimo (Markowitz): {m2['opt_return']:.2%}")
                context_parts.append(f"- Volatilidade Ótima: {m2['opt_vol']:.2%}")
                context_parts.append("- Pesos Ótimos: " + ", ".join(
                    f"{t}: {w:.1%}" for t, w in m2["optimal_weights"].items()
                ))

            if "beta_alpha" in m2:
                for ba in m2["beta_alpha"]:
                    context_parts.append(f"- {ba['Ticker']}: Beta={ba['Beta']}, Alpha={ba['Alpha (anual)']}")

            if "regression_r2" in m2:
                context_parts.append(f"- R² da Regressão (Port vs SPY): {m2['regression_r2']:.4f}")

            if "correlation" in m2:
                context_parts.append(f"- Correlação: {m2['correlation']}")

            collected_context = "\n".join(context_parts)

        # ── Agent 2: Analyst ──
        with st.spinner("🧠 Agent 2 — Gerando insights e análise..."):
            analyst_prompt = f"""Você é um analista financeiro sênior. Com base nos dados abaixo, gere uma análise detalhada com insights sobre:
1. Tendências de preço e qualidade das previsões LSTM
2. Perfil de risco-retorno do portfólio
3. Correlações e diversificação
4. Comparação com benchmark (SPY)
5. Pontos de atenção e oportunidades

Dados:
{collected_context}

Responda em português brasileiro, de forma técnica mas acessível a executivos."""

            try:
                if "OpenAI" in api_choice:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    analyst_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": analyst_prompt}],
                        temperature=0.4,
                        max_tokens=2000,
                    )
                    analyst_insights = analyst_response.choices[0].message.content
                else:
                    from groq import Groq
                    client = Groq(api_key=api_key)
                    analyst_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": analyst_prompt}],
                        temperature=0.4,
                        max_tokens=2000,
                    )
                    analyst_insights = analyst_response.choices[0].message.content
            except Exception as e:
                st.error(f"Erro no Agent 2 (Analyst): {e}")
                st.stop()

        # ── Agent 3: Reporter ──
        with st.spinner("📝 Agent 3 — Gerando relatório executivo..."):
            reporter_prompt = f"""Você é um especialista em relatórios financeiros executivos. 
Com base na análise do analista e nos dados originais, gere um relatório executivo completo em Markdown com:

1. **📋 Resumo Executivo** — visão geral em 3-5 linhas
2. **📊 Tabela de Recomendações** — formato Markdown:
   | Ticker | Preço Atual | Recomendação | Justificativa |
   Use emojis: 🟢 Comprar, 🟡 Manter, 🔴 Vender
3. **📈 Análise de Tendências** — baseada nas previsões LSTM
4. **⚖️ Análise de Risco do Portfólio** — Sharpe, Drawdown, Volatilidade
5. **🎯 Portfólio Ótimo** — pesos recomendados (Markowitz)
6. **💡 Insights e Oportunidades**
7. **⚠️ Riscos e Pontos de Atenção**
8. **🏁 Conclusão e Próximos Passos**

Análise do Analista:
{analyst_insights}

Dados Originais:
{collected_context}

Use formatação Markdown rica, emojis para visual appeal, e linguagem executiva em português brasileiro.
Seja objetivo, confiável e forneça justificativas claras para cada recomendação."""

            try:
                if "OpenAI" in api_choice:
                    reporter_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": reporter_prompt}],
                        temperature=0.3,
                        max_tokens=3000,
                    )
                    final_report = reporter_response.choices[0].message.content
                else:
                    reporter_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": reporter_prompt}],
                        temperature=0.3,
                        max_tokens=3000,
                    )
                    final_report = reporter_response.choices[0].message.content
            except Exception as e:
                st.error(f"Erro no Agent 3 (Reporter): {e}")
                st.stop()

        st.session_state.module3_report = final_report
        st.session_state.module3_done = True
        st.success("✅ Relatório executivo gerado com sucesso!")

# ── DISPLAY AI REPORT ──
if st.session_state.get("module3_done") and st.session_state.get("module3_report"):
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)
    section_header("📄", "Relatório Executivo — IA", "Gerado por multi-agentes")

    st.markdown(
        f'<div class="exec-card">{st.session_state.module3_report}</div>',
        unsafe_allow_html=True,
    )

    # Also render with native markdown for proper formatting
    with st.expander("📄 Relatório Completo (Markdown renderizado)", expanded=True):
        st.markdown(st.session_state.module3_report)

    # Download button
    st.download_button(
        label="⬇️ Download Relatório (.md)",
        data=st.session_state.module3_report,
        file_name=f"relatorio_executivo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """<div style="text-align:center; padding:20px; color:#999; font-size:0.8rem;">
        <strong>Finance Intelligence Suite</strong> · Powered by Yahoo Finance (yfinance) · LSTM Neural Networks · Markowitz Optimization<br>
        Desenvolvido pela Equipe de IA Financeira · 2026
    </div>""",
    unsafe_allow_html=True,
)
