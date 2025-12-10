
import math
from typing import Tuple, Dict
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return float('inf'), float('inf')
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = "call") -> float:
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    if option_type.lower().startswith('c'):
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = "call") -> Dict[str, float]:
    # edge cases
    if T <= 0:
        if option_type.lower().startswith('c'):
            delta = 1.0 if S > K else 0.0 if S < K else 0.5
            price = max(0.0, S - K)
        else:
            delta = -1.0 if S < K else 0.0 if S > K else -0.5
            price = max(0.0, K - S)
        return {"Price": price, "Delta": delta, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}

    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_minus_d1 = norm.cdf(-d1)
    cdf_minus_d2 = norm.cdf(-d2)

    sqrtT = math.sqrt(T)
    discounted_forward = S * math.exp(-q * T)

    # Price
    price = bs_price(S, K, r, q, sigma, T, option_type)

    # Delta
    if option_type.lower().startswith('c'):
        delta = math.exp(-q * T) * cdf_d1
    else:
        delta = math.exp(-q * T) * (cdf_d1 - 1.0)

    # Gamma (same for call & put)
    gamma = (math.exp(-q * T) * pdf_d1) / (S * sigma * sqrtT)

    # Vega (per 1.0 absolute vol)
    vega = S * math.exp(-q * T) * pdf_d1 * sqrtT

    # Theta (annualized -> convert to per day)
    if option_type.lower().startswith('c'):
        theta_annual = (-S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrtT) - r * K * math.exp(-r * T) * cdf_d2 + q * S * math.exp(-q * T) * cdf_d1
    else:
        theta_annual = (-S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrtT) + r * K * math.exp(-r * T) * cdf_minus_d2 - q * S * math.exp(-q * T) * cdf_minus_d1
    theta_per_day = theta_annual / 365.0

    # Rho (per 1.0 change in r)
    if option_type.lower().startswith('c'):
        rho = K * T * math.exp(-r * T) * cdf_d2
    else:
        rho = -K * T * math.exp(-r * T) * cdf_minus_d2

    return {"Price": price, "Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta_per_day, "Rho": rho}

# Implied volatility solver: find sigma such that bs_price(...) == market_price
def implied_vol(market_price: float, S: float, K: float, r: float, q: float, T: float, option_type: str = "call",
                sigma_bounds: Tuple[float, float] = (1e-6, 5.0), tol: float = 1e-6, maxiter: int = 100) -> float:
    """
    Solve for implied volatility using Brent's method.
    sigma_bounds: (low, high) initial bracketing bounds
    Returns implied volatility (decimal, e.g., 0.2)
    """
    if market_price <= 0:
        return 0.0

    # Define function f(sigma) = BS_price(sigma) - market_price
    def f(sigma):
        return bs_price(S, K, r, q, sigma, T, option_type) - market_price

    low, high = sigma_bounds
    # Make sure the bracket captures a sign change; widen if necessary
    f_low, f_high = f(low), f(high)
    # If f_low and f_high have same sign, try to expand high
    if f_low * f_high > 0:
        # expand high until sign change or until max high reached
        for _ in range(50):
            high *= 2.0
            f_high = f(high)
            if f_low * f_high <= 0:
                break
        else:
            raise ValueError("Could not bracket implied volatility. Try different bounds or check market price.")

    # Now use brentq
    iv = brentq(f, low, high, xtol=tol, maxiter=maxiter)
    return iv
