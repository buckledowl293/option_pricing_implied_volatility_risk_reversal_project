"""
iv_solver.py
============
Implied Volatility Solver using Newton-Raphson iteration with bisection fallback.

The Problem
-----------
Given an observed market price P_mkt for an option, find the volatility σ* such that:

    BS(S, K, r, T, σ*) = P_mkt

There is no closed-form inverse of the Black-Scholes formula with respect to σ.
However, the BS price is:
    - Smooth and continuous in σ
    - Strictly monotone increasing in σ (for both calls and puts)
    - Therefore has exactly one root for any valid market price

This makes it ideal for Newton-Raphson, which converges very quickly (quadratically)
once near the root.

Newton-Raphson Update Rule
--------------------------
    σ_{n+1} = σ_n − f(σ_n) / f'(σ_n)

where:
    f(σ)  = BS(σ) − P_mkt     (price error)
    f'(σ) = ∂BS/∂σ = Vega     (derivative = Vega!)

This is the elegant insight: Newton-Raphson for IV naturally uses the Vega
as its derivative term, making the two concepts directly linked.

Bisection Fallback
------------------
When vega is near zero (deep ITM/OTM options), Newton-Raphson becomes unreliable
because we're dividing by a very small number. In these cases, the solver falls
back to bisection, which is slower but guaranteed to converge (O(log n) steps)
given valid brackets.
"""

import numpy as np
import pandas as pd
from black_scholes import bs_price, vega_raw


# --------------------------------------------------------------------------- #
# Core Newton-Raphson Solver                                                   #
# --------------------------------------------------------------------------- #

def newton_raphson_iv(market_price, S, K, r, T, option_type='call',
                      sigma0=None, max_iter=200, tol=1e-7):
    """
    Compute implied volatility via Newton-Raphson iteration.

    Parameters
    ----------
    market_price : float - Observed market price (bid-ask mid)
    S            : float - Current spot price
    K            : float - Strike price
    r            : float - Risk-free rate (annualised, continuously compounded)
    T            : float - Time to expiry in years
    option_type  : str   - 'call' or 'put'
    sigma0       : float - Initial volatility guess (None = use auto-estimate)
    max_iter     : int   - Maximum Newton-Raphson iterations
    tol          : float - Convergence tolerance on |σ_{n+1} - σ_n|

    Returns
    -------
    float : Implied volatility (annualised decimal, e.g. 0.18 = 18%),
            or np.nan if the solver fails to converge.
    """
    # ------------------------------------------------------------------ #
    # Input Validation                                                     #
    # ------------------------------------------------------------------ #
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return np.nan

    discount = np.exp(-r * T)
    otype = option_type.lower()

    # No-arbitrage lower bound (intrinsic value)
    if otype == 'call':
        intrinsic = max(S - K * discount, 0.0)
        upper_bound = S                  # call can't exceed spot
    else:
        intrinsic = max(K * discount - S, 0.0)
        upper_bound = K * discount       # put can't exceed discounted strike

    # Market price outside no-arbitrage bounds → no valid IV exists
    # 5 cent tolerance
    if market_price < intrinsic - 0.05:
        return np.nan
    if market_price >= upper_bound:
        return np.nan

    # ------------------------------------------------------------------ #
    # Initial Guess                                                        #
    # ------------------------------------------------------------------ #
    if sigma0 is None:
        # Brenner-Subrahmanyam (1988) approximation for ATM options:
        #   σ ≈ (C/S) × √(2π/T)
        # starting point, will be refined quickly by N-R.
        bs_approx = np.sqrt(2 * np.pi / T) * (market_price / S)
        sigma = np.clip(bs_approx, 0.01, 3.0)
    else:
        sigma = float(sigma0)

    # ------------------------------------------------------------------ #
    # Newton-Raphson Iterations                                            #
    # ------------------------------------------------------------------ #
    for _ in range(max_iter):
        price  = bs_price(S, K, r, T, sigma, otype)
        v      = vega_raw(S, K, r, T, sigma)  # = ∂BS/∂σ

        price_error = price - market_price

        # Check for early convergence
        if abs(price_error) < 1e-9:
            return sigma

        # Vega near zero = N-R unstable = hand off to bisection
        if abs(v) < 1e-10:
            return _bisection_iv(market_price, S, K, r, T, otype)

        # Standard Newton-Raphson step
        sigma_new = sigma - price_error / v

        # Clamp to economically sensible range (0.1% to 500%)
        sigma_new = np.clip(sigma_new, 0.001, 5.0)

        if abs(sigma_new - sigma) < tol:
            return sigma_new

        sigma = sigma_new

    # N-R didn't converge — try bisection as final fallback

    if sigma == 0.001:
        return _bisection_iv(market_price, S, K, r, T, otype)
    if sigma > 4.9:
        return np.nan
    return sigma
    


# --------------------------------------------------------------------------- #
# Bisection Fallback                                                           #
# --------------------------------------------------------------------------- #

def _bisection_iv(market_price, S, K, r, T, option_type,
                  low=0.001, high=5.0, max_iter=100, tol=1e-6):
    """
    Bisection method for implied volatility.

    Slower than Newton-Raphson (linear vs quadratic convergence) but
    guaranteed to find the root given valid brackets [low, high].

    Because BS price is strictly increasing in σ, bisection is safe:
    we know there is exactly one crossing of the market price.
    """
    price_low  = bs_price(S, K, r, T, low,  option_type)
    price_high = bs_price(S, K, r, T, high, option_type)

    # If market price is outside our bracketing range, return NaN
    if market_price < price_low or market_price > price_high:
        return np.nan

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        price_mid = bs_price(S, K, r, T, mid, option_type)

        if abs(price_mid - market_price) < tol or (high - low) < tol:
            return mid

        if price_mid < market_price:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0   # best estimate after max iterations


# --------------------------------------------------------------------------- #
# Robust wrapper: try multiple initial guesses                                 #
# --------------------------------------------------------------------------- #

def implied_vol(market_price, S, K, r, T, option_type='call'):
    """
    Compute implied volatility robustly.

    Tries Newton-Raphson with multiple initial guesses; falls back to
    bisection if all attempts fail. Returns np.nan only if the price is
    genuinely outside no-arbitrage bounds.

    This is the recommended entry point for general use.
    """
    # Try Newton-Raphson with several different starting points
    for sigma0 in [0.20, 0.15, 0.30, 0.50, 0.80]:
        iv = newton_raphson_iv(market_price, S, K, r, T, option_type,
                               sigma0=sigma0)
        if not np.isnan(iv):
            return iv

    # Final fallback: pure bisection
    return _bisection_iv(market_price, S, K, r, T, option_type.lower())


# --------------------------------------------------------------------------- #
# Vectorised computation for a full options chain DataFrame                   #
# --------------------------------------------------------------------------- #

def compute_iv_for_chain(options_df, S, r, option_type = "call"):

    ####### nnnnnnnnnnnnnnnneeeeeeeeeeeeeeeeeeeeeeedddddddddddddddddddsssssssssss fixing
    """
    Compute implied volatility for every row in an options chain DataFrame.

    Parameters
    ----------
    options_df   : pd.DataFrame - Must have columns ['strike', ###'T'###, price_column]
    S            : float        - Spot price
    r            : float        - Risk-free rate
    option_type  : str          - 'call' or 'put'
    price_column : str          - Column name containing market prices

    Returns
    -------
    pd.Series of implied volatilities, same index as options_df.
    Failed computations are np.nan.
    """
    #use a for loop since newton raphson too tricky to vectorize
    ivs = []
    for _, row in options_df.iterrows():
        iv = implied_vol(
            market_price=row["price_column"],
            S=S,
            K=row['strike'],
            r=r,
            T=row['T'],
            option_type=option_type
        )
        ivs.append(iv)
    
    
    return pd.Series(ivs, index=options_df.index, name='iv')