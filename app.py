import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_price = 100
mu = 0.07
sigma = 0.2
time_horizon = 1
investment_amount = 1200
periods = 12
simulations = 1000

# GBM Simulation Function
def simulate_gbm(S0, mu, sigma, T, steps, sims):
    dt = T / steps
    prices = np.zeros((steps + 1, sims))
    prices[0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(sims)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices

# Simulate Price Paths
steps = periods
price_paths = simulate_gbm(initial_price, mu, sigma, time_horizon, steps, simulations)

# Strategies
dca_results = []
lsi_results = []

for sim in range(simulations):
    prices = price_paths[:, sim]
    
    # DCA
    dca_investments = investment_amount / periods
    dca_shares = np.sum(dca_investments / prices[:periods])
    dca_results.append(dca_shares * prices[-1])
    
    # LSI
    lsi_shares = investment_amount / prices[0]
    lsi_results.append(lsi_shares * prices[-1])

# Analysis
dca_mean = np.mean(dca_results)
lsi_mean = np.mean(lsi_results)
dca_std = np.std(dca_results)
lsi_std = np.std(lsi_results)

# Streamlit App
st.title("DCA vs LSI Simulation")
st.write("""This app simulates and compares Dollar-Cost Averaging (DCA) and Lump Sum Investing (LSI) strategies using Geometric Brownian Motion.""")

# Display Results
st.subheader("Results")
st.write(f"DCA Mean Final Value: ${dca_mean:.2f}, Standard Deviation: ${dca_std:.2f}")
st.write(f"LSI Mean Final Value: ${lsi_mean:.2f}, Standard Deviation: ${lsi_std:.2f}")

# Plot Histograms
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(dca_results, bins=50, alpha=0.6, label='DCA Final Values')
ax.hist(lsi_results, bins=50, alpha=0.6, label='LSI Final Values')
ax.axvline(dca_mean, color='blue', linestyle='dashed', linewidth=1, label='DCA Mean')
ax.axvline(lsi_mean, color='orange', linestyle='dashed', linewidth=1, label='LSI Mean')
ax.set_title('DCA vs LSI Final Portfolio Values')
ax.set_xlabel('Portfolio Value ($)')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)
