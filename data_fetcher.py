
#fetch the options chains from using yfinance 

import yfinance as yf
import numpy as np
import pandas as pd
import datetime

'''
in yfinance the expiry date T of a contract is embedded into its contractSymbol
e.g AAPL260717C00180000 where AAPL is the ticker, the date goes: year-month-day in this case the date is 17th july 2026
the letter C or P is the type of option (call or put)
number at the end is the strike price K of the option, in this case it is $180.00 

also need to get risk free interest rate (use ^IRX ticker) + spot price S of the asset

since option chains are split by there expiry date I want to take every chain and then combine it into one big master data frame


'''
#first fetch the options data and expiry dates
def get_all_expiry_dates(tickname  = "SPY"):

    ticker = yf.Ticker(tickname)
    xpirations = ticker.options


    return xpirations

def get_combined_option_chain(tickname , xpirations , spot_price, type):
    all_calls_list = []
    ticker = yf.Ticker(tickname)
    #add all 2 a list and then combine from list
    for dates in xpirations:
        if type == "call":
            current_df = ticker.option_chain(dates).calls
            all_calls_list.append(current_df)
        else:
            current_df = ticker.option_chain(dates).puts
            all_calls_list.append(current_df)

#then combine all in master dataframe

    option_df = pd.concat(all_calls_list, ignore_index=True)
    '''
    during testing there was a problem with short term trades giving extremely high implied volatility 
    and with calls with low trade volume and openInterest as well as bid price of around 0
    next part is just cleaning the option trade
    '''

    #--------------------------------------------------------------
    # Keep only options with active trading volume or outstanding contracts
    # or get rid of calls with no bids
    option_df = option_df[(option_df["volume"] > 5)]
    #option_df = option_df[option_df['bid'] > 0.05]
    #then restrict the data so that it is within a 30% range of current value
    lower_bound = spot_price * 0.70
    upper_bound = spot_price * 1.30
    

    #option_df = option_df[(option_df["strike"] >= lower_bound) & (option_df["strike"] <= upper_bound)]

    #--------------------------------------------------------------

    return option_df







#make a function that returns the option data with the expiration date as a new column
def add_expiry_dateT_col(option_df,tickname,spot_price):
    # first fetch the contract symbols for the option chain for todays date

    option_contractSymbol = option_df["contractSymbol"]
    name_length = len(tickname)
    # then get the date from that and calculate the difference from todays date in years.
    # write an algorithm to find difference in dates
    # this breaks if the dataframe is not a .calls or .puts one 
    option_df["expiry_dates"] = pd.to_datetime(
    option_df['contractSymbol'].str[name_length:name_length+6], 
    format = '%y%m%d'
    )
    current_date = datetime.datetime.now()
    option_df["T"] = (option_df["expiry_dates"].dt.year - current_date.year) + (option_df["expiry_dates"].dt.month - current_date.month)/12 + (option_df["expiry_dates"].dt.day - current_date.day)/365
    '''
    now clean data with low T values since when out the money skews the implied volatility significantly
    '''
    # Create a mask to identify ultra-short dated options
    short_dated = option_df['T'] < 0.1

# Define a tight strike window for short-dated options (e.g., within 5% of spot)
    atm_window = (option_df['strike'] >= spot_price * 0.95) & (option_df['strike'] <= spot_price * 1.05)

# Keep the row if it's long-dated OR if it's short-dated but strictly ATM
    option_df = option_df[(~short_dated) | (short_dated & atm_window)]
    

    
    


    return option_df

#also need to add a price column for options to find out how much option contract should cost
#do this by finding midpoint of bid and ask
def add_price_col(option_df):


    option_df["price_column"] = (option_df["bid"] + option_df["ask"])/2

    return option_df

#also need to find the spot price and the interest rate
def find_spot_price(tickname = "SPY"):
    ticker = yf.Ticker(tickname)

#Fetch the most recent 1 day of data at a 1-minute interval (for live prices)
    data = ticker.history(period="1d", interval="1m")

# Get the very last available price
    if not data.empty:
        spot_price = data['Close'].iloc[-1]
    
    else:
    # Fallback to daily data if the market is closed or 1m interval is empty
        data_daily = ticker.history(period="1d")
        spot_price = data_daily['Close'].iloc[-1]

    return spot_price



def get_risk_free_rate():
    
    irx_ticker = yf.Ticker("^IRX")

# 2. Grab the latest intraday history
# (Using 1d period with 1m interval to get the live market quote)
    live_data = irx_ticker.history(period="1d", interval="1m")

    if not live_data.empty:
    # yfinance quotes yields multiplied by 10, so divide by 10 to get the percentage
        interest = live_data['Close'].iloc[-1] / 10
        
    else:
    # Fallback if market is fully closed and 1m interval is blank
        daily_data = irx_ticker.history(period="1d")
        interest = daily_data['Close'].iloc[-1] / 10

#convert to a decimal from percentage when returned so can be used
    interest = interest / 100

    return interest