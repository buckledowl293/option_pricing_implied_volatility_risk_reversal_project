import yfinance as yf
from  data_fetcher import *
from implied_volatility import *
from plotplotplot import *
from risk_reversal import get_risk_reversal

def get_plot(tickname,type):

    r = get_risk_free_rate()
    spot = find_spot_price(tickname)
    expirations = get_all_expiry_dates(tickname)
    option_df = get_combined_option_chain(tickname,expirations,spot,type)
    new_option_df = add_expiry_dateT_col(option_df,tickname,spot)
    new_option_df = add_price_col(new_option_df)

    iv = compute_iv_for_chain(new_option_df,spot,r,tickname)

    plot_implied_volatility_surface2(iv,new_option_df)

    return 

def get_clean_chain(tickname,type):


    r = get_risk_free_rate()
    spot = find_spot_price(tickname)
    expirations = get_all_expiry_dates(tickname)
    option_df = get_combined_option_chain(tickname,expirations,spot,type)
    
    option_df = add_expiry_dateT_col(option_df,tickname,spot)
    option_df = add_price_col(option_df)

    
    
    
    return option_df

def stock_risk_reversal(tickname):

    calls = get_clean_chain(tickname,"call")
    puts = get_clean_chain(tickname,"put")
    risk_reversal = get_risk_reversal(calls,puts,tickname)

    return risk_reversal

#get_clean_chain("GOOG","call")
print(stock_risk_reversal("GOOG"))

#get_plot("AAPL","put")


