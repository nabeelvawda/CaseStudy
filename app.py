import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="BTC Signal Predictor", layout="wide")

# ── Load artifacts (cached so they only load once per session) ──────────
@st.cache_resource
def load_model():
    model = joblib.load("ma_ema_model.joblib")
    scaler = joblib.load("ma_ema_scaler.joblib")
    return model, scaler

@st.cache_data
def load_history():
    return pd.read_csv("ma_ema_history.csv", parse_dates=["signaled_at"])

@st.cache_data
def load_model_summary():
    return pd.read_csv("model_summary.csv")

model, scaler = load_model()
history = load_history()
summary = load_model_summary()

st.title("BTC/USDT Signal Predictor")
st.caption(
    "Predicts 24h-forward price direction from a moving-average indicator, "
    "trained on BasicMovingAverageBot / ExponentialMovingAverageBot signals "
    "(Feb 2025 – May 2026)."
)

tab_predict, tab_explore = st.tabs(["Predict", "Explore the data"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Try a prediction")

    col_input, col_result = st.columns([1, 1.4])

    with col_input:
        symbol = st.selectbox("Symbol", ["BTC/USDT"], disabled=True,
                               help="Model was trained on BTC/USDT only — see README for why.")
        current_price = st.number_input("Current price (USD)", min_value=0.0,
                                         value=98000.0, step=100.0)
        indicator_value = st.number_input("Indicator value (moving average, USD)",
                                           min_value=0.0, value=97500.0, step=100.0)
        bot_type = st.radio("Moving average type", ["Basic", "Exponential"], horizontal=True)

        run = st.button("Predict", type="primary", use_container_width=True)

    with col_result:
        if run:
            pct_diff = (current_price - indicator_value) / indicator_value
            bot_type_encoded = 1 if bot_type == "Exponential" else 0

            X = pd.DataFrame([{
                "pct_diff_from_ma": pct_diff,
                "bot_type_encoded": bot_type_encoded,
            }])
            X_scaled = scaler.transform(X)

            proba_up = model.predict_proba(X_scaled)[0, 1]
            model_pred = "up" if proba_up >= 0.5 else "down"
            model_conf = proba_up if model_pred == "up" else 1 - proba_up

            # "Follow the bot" baseline — verified from historical data:
            # Buy (price > MA) / Sell (price < MA) is effectively a clean
            # zero threshold on pct_diff_from_ma (see README for the check).
            bot_signal = "Buy" if pct_diff > 0 else "Sell"
            bot_pred = "up" if bot_signal == "Buy" else "down"

            st.markdown("#### Trained model")
            m1, m2 = st.columns(2)
            m1.metric("Prediction", model_pred.upper())
            m2.metric("Confidence", f"{model_conf:.1%}")
            st.caption(
                f"pct. diff from MA: {pct_diff:+.2%}  ·  "
                f"P(up) = {proba_up:.3f}"
            )

            st.markdown("#### Bot's own logic (baseline)")
            b1, b2 = st.columns(2)
            b1.metric("Bot signal", bot_signal)
            b2.metric("Implied direction", bot_pred.upper())

            if model_pred != bot_pred:
                st.warning(
                    "The trained model and the bot's own rule disagree here. "
                    "Per the evaluation (see Explore tab), the bot's own logic "
                    "outperformed the trained model in the post-rollout period — "
                    "worth weighing that when reading the model's output."
                )
            else:
                st.info("Model and bot logic agree on direction for this input.")
        else:
            st.info("Enter values and click Predict to see results.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("Signal history")
    st.line_chart(
        history.set_index("signaled_at")[["current_price", "moving_average"]]
    )

    st.subheader("Feature distribution: % diff from moving average")
    st.bar_chart(
        np.histogram(history["pct_diff_from_ma"].dropna(), bins=30)[0]
    )

    st.subheader("Model vs. baselines — evaluation summary")
    st.dataframe(summary, use_container_width=True)
    st.caption(
        "The trained model performs close to chance (ROC-AUC ≈ 0.49 pre-rollout). "
        "The 'follow the bot' baseline outperformed the trained model in the "
        "post-rollout period. Full reasoning and diagnosis in the README."
    )
