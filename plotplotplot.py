
'''
now need to plot all of the data using matplotlib
fetch the relevant information from the data_fetcher file:
get r, S, T, K and implied volatility sigma from all options in data frame
plot implied volatility against K and T to give 3D plot


'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_implied_volatility_surface(implied_volatility_series, options_df):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

# X = Strike, Y = Time to Expiration, Z = Implied Volatility
    x = options_df['strike']
    y = options_df['T']
    z = implied_volatility_series

# Plot the 3D triangular surface
    surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)

# Add a color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')

# Labels and view angle
    ax.set_title('Implied Volatility Surface', fontsize=14)
    ax.set_xlabel('Strike Price', fontsize=11)
    ax.set_ylabel('Time to Expiration (T in Years)', fontsize=11)
    ax.set_zlabel('Implied Volatility (IV)', fontsize=11)

# Adjust the viewing angle to see the curve clearly (elevation, azimuth)
    ax.view_init(elev=20, azim=-45)

    plt.show()
    return

def plot_implied_volatility_surface2(implied_volatility_series, options_df):

    # 1. Create a dense, uniform grid of strikes and expiries
    strike_grid = np.linspace(options_df['strike'].min(), options_df['strike'].max(), 100)
    T_grid = np.linspace(options_df['T'].min(), options_df['T'].max(), 100)
    X, Y = np.meshgrid(strike_grid, T_grid)

    # 2. Interpolate your calculated IVs onto this perfect grid
    Z = griddata(
        points=(options_df['strike'], options_df['T']), 
        values=implied_volatility_series, 
        xi=(X, Y), 
        method='linear'  # 'cubic' creates an even smoother look if data is clean
    )

    # 3. Plot using plot_surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Expiration (T)')
    ax.set_zlabel('Implied Volatility')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()





    return
