import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def calculate_portfolio_stats(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_stddev
    return portfolio_return, portfolio_stddev, sharpe_ratio

def generate_random_weights(num_assets):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    return weights

def plot_efficient_frontier(results):
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
    plt.title('Fronteira Eficiente')
    plt.xlabel('Risco (Desvio Padrão)')
    plt.ylabel('Retorno Esperado')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def find_optimal_portfolio(results):
    max_sharpe_idx = np.argmax(results[2,:])
    sdp, rp = results[1,max_sharpe_idx], results[0,max_sharpe_idx]
    return sdp, rp

def print_portfolio_info(rp, sdp, optimal_weights):
    pesos_np = np.round(optimal_weights, 4)
    print(f'Retorno do portfólio ótimo: {round(rp,4)}')
    print(f'Risco do portfólio ótimo: {round(sdp,4)}')
    print(f'Pesos do portfólio ótimo: {pesos_np}')

def main():
    tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
    returns = download_data(tickers, "2020-01-01", "2023-01-01")
    mean_returns, cov_matrix = calculate_portfolio_stats(returns)

    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = generate_random_weights(len(tickers))
        portfolio_return, portfolio_stddev, sharpe_ratio = calculate_portfolio_performance(weights, mean_returns, cov_matrix)

        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = sharpe_ratio
        
    plot_efficient_frontier(results)

    sdp, rp = find_optimal_portfolio(results)
    optimal_weights = generate_random_weights(len(tickers))
    print_portfolio_info(rp, sdp, optimal_weights)

if __name__ == "__main__":
    main()



    