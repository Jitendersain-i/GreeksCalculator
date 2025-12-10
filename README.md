# Options Greeks Calculator

A Python Streamlit app to calculate **Black-Scholes option price and Greeks**, visualize them, and compute implied volatility.

## Features
- Compute Price, Delta, Gamma, Vega, Theta, Rho for call and put options
- Interactive visualizations of Price and Greeks vs underlying price
- Compute implied volatility from market option price
- Export CSV of computed series
- Streamlit GUI

## Installation (Python)
```bash
git clone <your_repo_url>
cd GreeksCalculator
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
