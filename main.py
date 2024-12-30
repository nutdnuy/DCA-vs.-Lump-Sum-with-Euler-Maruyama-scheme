import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_price = 100       # Starting stock price (S0)
mu = 0.07                 # Expected annualized return
sigma = 0.2               # Annualized volatility
time_horizon = 1          # Time horizon in years
investment_amount = 1200  # Total investment
periods = 12              # Number of DCA periods
simulations = 1000        # Number of simulated paths

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

print(f"DCA Mean Final Value: ${dca_mean:.2f}, Standard Deviation: ${dca_std:.2f}")
print(f"LSI Mean Final Value: ${lsi_mean:.2f}, Standard Deviation: ${lsi_std:.2f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.hist(dca_results, bins=50, alpha=0.6, label='DCA Final Values')
plt.hist(lsi_results, bins=50, alpha=0.6, label='LSI Final Values')
plt.axvline(dca_mean, color='blue', linestyle='dashed', linewidth=1, label='DCA Mean')
plt.axvline(lsi_mean, color='orange', linestyle='dashed', linewidth=1, label='LSI Mean')
plt.title('DCA vs LSI Final Portfolio Values')
plt.xlabel('Portfolio Value ($)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
