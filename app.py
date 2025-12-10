
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from options_greeks import greeks, bs_price, implied_vol

st.set_page_config(page_title="Options Greeks Calculator", layout="wide")

st.title("Options Greeks Calculator — Black–Scholes")
st.markdown("""
Enter option parameters, compute price, greeks, implied volatility (from market price),
and visualize Price / Delta / Gamma / Vega vs Underlying price.
""")

# ---------------- Input column ----------------
with st.sidebar:
    st.header("Option Inputs")
    S = st.number_input("Underlying price (S)", value=100.0, min_value=0.0001, step=1.0, format="%.4f")
    K = st.number_input("Strike price (K)", value=100.0, min_value=0.0001, step=1.0, format="%.4f")
    r_pct = st.number_input("Risk-free rate r (annual %, e.g., 5)", value=5.0, step=0.1)
    q_pct = st.number_input("Dividend yield q (annual %, e.g., 0)", value=0.0, step=0.1)
    sigma_pct = st.number_input("Volatility sigma (annual %, e.g., 20)", value=20.0, step=0.5)
    T_days = st.number_input("Time to maturity (days)", value=180, min_value=0)
    opt_type = st.selectbox("Option type", ("call", "put"))
    market_price = st.number_input("Market option price (optional) — to compute implied vol", value=0.0, step=0.1)
    run_button = st.button("Compute")

# Convert inputs
r = r_pct / 100.0
q = q_pct / 100.0
sigma = sigma_pct / 100.0
T = max(T_days, 0) / 365.0

# Live compute or on button
if run_button:
    try:
        # Compute price & greeks
        res = greeks(S, K, r, q, sigma, T, opt_type)
        price = res["Price"]
        st.subheader("Black–Scholes Results")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.metric("Model Price", f"{price:,.6f}")
            st.metric("Delta", f"{res['Delta']:.6f}")
            st.metric("Gamma", f"{res['Gamma']:.6f}")
        with col2:
            st.metric("Vega (per 1.0 vol)", f"{res['Vega']:.6f}")
            st.metric("Theta (per day)", f"{res['Theta']:.6f}")
            st.metric("Rho", f"{res['Rho']:.6f}")

        # Implied volatility if market price supplied
        if market_price and market_price > 0:
            try:
                iv = implied_vol(market_price, S, K, r, q, T, opt_type)
                st.success(f"Implied Volatility (annual) = {iv*100:.4f}%")
            except Exception as e:
                st.error(f"Implied vol error: {e}")

        # Plots: Price vs S and Greeks vs S
        st.subheader("Visualizations")
        # Choose range around S
        pct = st.slider("Range around S (± %)", 10, 200, 50)
        low = max(0.01, S * (1 - pct/100.0))
        high = S * (1 + pct/100.0)
        n_points = st.slider("Number of points", 50, 1000, 200)
        S_range = np.linspace(low, high, n_points)

        prices = [bs_price(s, K, r, q, sigma, T, opt_type) for s in S_range]
        deltas = [greeks(s, K, r, q, sigma, T, opt_type)["Delta"] for s in S_range]
        gammas = [greeks(s, K, r, q, sigma, T, opt_type)["Gamma"] for s in S_range]
        vegas = [greeks(s, K, r, q, sigma, T, opt_type)["Vega"] for s in S_range]

        # Price plot
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=S_range, y=prices, mode='lines', name='BS Price'))
        fig_price.add_trace(go.Scatter(x=[S], y=[price], mode='markers', name='Current S', marker=dict(size=8)))
        fig_price.update_layout(title="Option Price vs Underlying Price", xaxis_title="Underlying Price S", yaxis_title="Option Price")
        st.plotly_chart(fig_price, use_container_width=True)

        # Greeks plot (Delta, Gamma, Vega)
        fig_greek = go.Figure()
        fig_greek.add_trace(go.Scatter(x=S_range, y=deltas, mode='lines', name='Delta'))
        fig_greek.add_trace(go.Scatter(x=S_range, y=gammas, mode='lines', name='Gamma'))
        fig_greek.add_trace(go.Scatter(x=S_range, y=vegas, mode='lines', name='Vega (per 1.0 vol)'))
        fig_greek.update_layout(title="Greeks vs Underlying Price", xaxis_title="Underlying Price S", yaxis_title="Value")
        st.plotly_chart(fig_greek, use_container_width=True)

        # Tabular output of sample points
        sample_df = pd.DataFrame({"S": S_range, "Price": prices, "Delta": deltas, "Gamma": gammas, "Vega": vegas})
        st.dataframe(sample_df.head(10))

        # Download CSV
        csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV of series", data=csv, file_name="greeks_series.csv", mime="text/csv")

    except Exception as exc:
        st.error(f"Error computing results: {exc}")
else:
    st.info("Fill inputs on the left and press **Compute** to see results.")
