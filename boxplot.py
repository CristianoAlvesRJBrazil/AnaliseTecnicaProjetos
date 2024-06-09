# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Definindo os símbolos das ações
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Obtendo dados de preços de fechamento ajustados
data = yf.download(tickers, start='2023-01-01', end='2023-12-31')['Adj Close']

# Exibindo as primeiras linhas dos dados
print(data.head())


# Cálculo dos retornos diários
returns = data.pct_change().dropna()

# Exibindo as primeiras linhas dos retornos
print(returns.head())

# Criação de um boxplot para os retornos diários
plt.figure(figsize=(10, 6))
sns.boxplot(data=returns)
plt.title('Distribuição dos Retornos Diários das Ações')
plt.ylabel('Retorno Diário')
plt.xlabel('Ação')
plt.show()

# Identificação e destaque de outliers nos retornos
plt.figure(figsize=(10, 6))
sns.boxplot(data=returns)
plt.title('Outliers nos Retornos Diários das Ações')
plt.ylabel('Retorno Diário')
plt.xlabel('Ação')

# Anotação dos outliers
for i in range(returns.shape[1]):
    data_series = returns.iloc[:, i]
    outliers = data_series[(data_series < data_series.quantile(0.25) - 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25))) |
                           (data_series > data_series.quantile(0.75) + 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25)))]
    for outlier in outliers:
        plt.text(i, outlier, round(outlier, 4), horizontalalignment='center', color='red')

plt.show()

# Definindo os símbolos das ações por setor
tickers_sectors = {
    'Tecnologia': ['AAPL', 'MSFT', 'GOOGL'],
    'Saúde': ['JNJ', 'PFE', 'MRK'],
    'Finanças': ['JPM', 'BAC', 'WFC']
}

# Obtenção e preparação dos dados por setor
returns_sectors = {}

for sector, symbols in tickers_sectors.items():
    data_sector = yf.download(symbols, start='2023-01-01', end='2023-12-31')['Adj Close']
    returns_sector = data_sector.pct_change().dropna()
    returns_sectors[sector] = returns_sector.mean(axis=1)

# Criação de um DataFrame consolidado
df_returns_sectors = pd.DataFrame(returns_sectors)

# Criação de boxplots para os setores
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_returns_sectors)
plt.title('Distribuição dos Retornos Diários por Setor')
plt.ylabel('Retorno Diário')
plt.xlabel('Setor')
plt.show()
