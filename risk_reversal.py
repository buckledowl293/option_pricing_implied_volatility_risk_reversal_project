
'''
next find 25 delta risk reversal
in order to do this need to calculate the delta for every single strike in the chain
then interpolate data to find the strike price where delta = 0.25 for calls and -0.25 for puts

find the implied volatility at each of these values and subtract calls IV from puts IV to give risk reversal

'''

from black_scholes import delta
from data_fetcher import *
from implied_volatility import compute_iv_for_chain
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.interpolate import interp1d

def get_risk_reversal(options_df_calls,options_df_puts,tickname):

    
    r = get_risk_free_rate()
    spot = find_spot_price(tickname)

    options_df_calls["iv"] = compute_iv_for_chain(options_df_calls,spot,r,"call")
    options_df_puts["iv"] = compute_iv_for_chain(options_df_puts,spot,r,"put")

    options_df_calls["delta"] = options_df_calls.apply(lambda row: delta(spot, row["strike"], r, row["T"], row["iv"], "call"), axis=1)
    options_df_puts["delta"] = options_df_puts.apply(lambda row: delta(spot, row["strike"], r, row["T"], row["iv"], "put"), axis=1)

    # Drop duplicates or non-monotonic values for interpolation safely
    options_df_calls = options_df_calls.sort_values("delta")
    options_df_puts = options_df_puts.sort_values("delta")

    #run the interpolation
    try:
        call_interp = interp1d(options_df_calls['delta'], options_df_calls['impliedVolatility'], kind='linear', fill_value="extrapolate")
        call_iv_25d = float(call_interp(0.25))
    except Exception:
        return None

    try:
        put_interp = interp1d(options_df_puts['delta'], options_df_puts['impliedVolatility'], kind='linear', fill_value="extrapolate")
        put_iv_25d = float(put_interp(-0.25))
    except Exception:
        return None
    
    #calculate risk reversal
    risk_reversal = call_iv_25d - put_iv_25d

    return risk_reversal

