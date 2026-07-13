"""
black_scholes.py
================
Black-Scholes (1973) option pricing model for European options.

Implements the closed-form formulas for call and put prices, and all five
standard Greeks: Delta, Gamma, Vega, Theta, Rho.

Reference: Black, F. & Scholes, M. (1973). "The Pricing of Options and
Corporate Liabilities." Journal of Political Economy.

Mathematical background:
    Given:
        S     = current spot price
        K     = strike price
        r     = continuously compounded risk-free rate (annualised)
        T     = time to expiry (in years)
        sigma = annualised volatility of the underlying

    Auxiliary quantities:
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d2 = d1 - σ√T

    Prices:
        Call = S·N(d1) - K·e^{-rT}·N(d2)
        Put  = K·e^{-rT}·N(-d2) - S·N(-d1)

    where N(·) is the standard normal CDF.
"""

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Core d1 / d2 computation                                                    #
# --------------------------------------------------------------------------- #

def _d1_d2(S, K, r, T, sigma):
    """
    Compute the d1 and d2 quantities used throughout the BS formulae.

    Parameters
    ----------
    S     : float  - Spot price of the underlying
    K     : float  - Strike price
    r     : float  - Risk-free rate (annualised, continuously compounded)
    T     : float  - Time to expiry in years
    sigma : float  - Volatility (annualised)

    Returns
    -------
    (d1, d2) : tuple of floats
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# --------------------------------------------------------------------------- #
# Option Price                                                                 #
# --------------------------------------------------------------------------- #

def bs_price(S, K, r, T, sigma, option_type='call'):
    """
    Black-Scholes price for a European call or put option.

    Parameters
    ----------
    S           : float - Spot price
    K           : float - Strike price
    r           : float - Risk-free rate (annualised)
    T           : float - Time to expiry (years)
    sigma       : float - Volatility (annualised)
    option_type : str   - 'call' or 'put'

    Returns
    -------
    float : Theoretical option price

    Notes
    -----
    At expiry (T=0): returns intrinsic value max(S-K, 0) for calls,
    max(K-S, 0) for puts.
    At zero vol (sigma=0): returns discounted intrinsic value.
    """
    option_type = option_type.lower()

    # --- Boundary conditions ---
    if T <= 0:
        return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)

    if sigma <= 0:
        fwd = S - K * np.exp(-r * T)
        return max(fwd, 0.0) if option_type == 'call' else max(-fwd, 0.0)

    # --- Standard BS formula ---
    d1, d2 = _d1_d2(S, K, r, T, sigma)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# --------------------------------------------------------------------------- #
# Greeks                                                                       #
# --------------------------------------------------------------------------- #

def delta(S, K, r, T, sigma, option_type='call'):
    """
    Delta (Δ): sensitivity of option price to spot price.

        ∂V/∂S

    Call delta ∈ [0, 1].   ATM call ≈ 0.50.
    Put  delta ∈ [-1, 0].  ATM put  ≈ -0.50.

    Interpretation: a delta of 0.60 means the option gains ~$0.60
    for every $1.00 rise in the underlying.

    Also interpretable as the risk-neutral probability that the
    option expires in-the-money (for calls).
    """
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, r, T, sigma)
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1.0


def gamma(S, K, r, T, sigma):
    """
    Gamma (Γ): rate of change of Delta with respect to spot price.

        ∂²V/∂S²  =  N'(d1) / (S·σ·√T)

    Identical for calls and puts. Always non-negative.

    Interpretation: high gamma means delta changes rapidly as spot
    moves — important for determining how frequently a delta-neutral
    position must be rebalanced. Gamma is highest for at-the-money
    options approaching expiry.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, T, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, r, T, sigma):
    """
    Vega (ν): sensitivity of option price to implied volatility.

        ∂V/∂σ  =  S·N'(d1)·√T

    Returned per 1 percentage-point change in volatility (÷100).
    Identical for calls and puts. Always non-negative.

    Interpretation: vega = 0.25 means the option gains $0.25 for every
    1% rise in implied volatility (vol up 18% → 19% = +$0.25).

    Note: internally, vega_raw() returns the unscaled derivative used
    by the Newton-Raphson IV solver.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, T, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0   # per 1% of vol


def vega_raw(S, K, r, T, sigma):
    """
    Raw Vega (∂V/∂σ, not scaled by 1/100).

    Used internally by the Newton-Raphson implied-volatility solver as
    the derivative in the update step:
        σ_{n+1} = σ_n − [BS(σ_n) − market_price] / vega_raw(σ_n)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, T, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, r, T, sigma, option_type='call'):
    """
    Theta (Θ): rate of change of option price with respect to time.

        ∂V/∂t

    Returned per calendar day (annualised value ÷ 365).
    Almost always negative — options lose value as time passes.

    Interpretation: theta = −0.05 means the option loses ~$0.05 per
    calendar day due to time decay alone, all else equal.

    Note: theta is the flip side of gamma — short-dated ATM options have
    the most negative theta but the most positive gamma.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, d2 = _d1_d2(S, K, r, T, sigma)
    time_decay = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

    if option_type == 'call':
        rate_term = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        rate_term = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return (time_decay + rate_term) / 365.0   # per calendar day


def rho(S, K, r, T, sigma, option_type='call'):
    """
    Rho (ρ): sensitivity of option price to the risk-free interest rate.

        ∂V/∂r

    Returned per 1 percentage-point change in rate (÷100).
    Calls have positive rho; puts have negative rho.

    Interpretation: rho = 0.10 means the call gains $0.10 for every
    1% rise in the risk-free rate. (Less important than other Greeks
    for short-dated equity options, more important for long-dated ones.)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    _, d2 = _d1_d2(S, K, r, T, sigma)

    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


# --------------------------------------------------------------------------- #
# Convenience: all at once                                                     #
# --------------------------------------------------------------------------- #

def all_greeks(S, K, r, T, sigma, option_type='call'):
    """
    Compute Black-Scholes price and all five Greeks in a single call.

    Returns
    -------
    dict with keys: price, delta, gamma, vega, theta, rho
    """
    return {
        'price': bs_price(S, K, r, T, sigma, option_type),
        'delta': delta(S, K, r, T, sigma, option_type),
        'gamma': gamma(S, K, r, T, sigma),
        'vega' : vega(S, K, r, T, sigma),       # per 1% vol change
        'theta': theta(S, K, r, T, sigma, option_type),  # per calendar day
        'rho'  : rho(S, K, r, T, sigma, option_type),    # per 1% rate change
    }