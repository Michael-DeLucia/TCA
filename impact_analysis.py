import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def kyles_lambda(trade_data):
    """
    Calculate Kyle's Lambda.

    Parameters:
    - trade_data: DataFrame with columns ['Time', 'Trade_Price', 'Trade_Volume']

    Returns:
    - lambda_value: Kyle's Lambda value.
    """
    trade_data['Return'] = trade_data['Trade_Price'].pct_change()
    trade_data.dropna(inplace=True)
    
    X = trade_data['Trade_Volume'].values.reshape(-1, 1)
    y = trade_data['Return'].values
    
    model = LinearRegression().fit(X, y)
    kyles_lambda = model.coef_[0]
    
    return kyles_lambda

def amihuds_illiquidity(trade_data):
    """
    Calculate Amihud's Illiquidity Ratio.

    Parameters:
    - trade_data: DataFrame with columns ['Time', 'Trade_Price', 'Trade_Volume']

    Returns:
    - illiquidity_ratio: Amihud's Illiquidity Ratio.
    """
    trade_data['Return'] = trade_data['Trade_Price'].pct_change()
    trade_data.dropna(inplace=True)
    
    illiquidity_ratio = np.mean(np.abs(trade_data['Return']) / trade_data['Trade_Volume'])
    
    return illiquidity_ratio

def market_impact_analysis(trade_data, liquidity_data):
    """
    Perform market impact analysis using Kyle's Lambda and Amihud's Illiquidity Ratio.

    Parameters:
    - trade_data: DataFrame with columns ['Time', 'Trade_Price', 'Trade_Volume']
    - liquidity_data: DataFrame with columns ['Time', 'Bid_Price', 'Ask_Price', 'Bid_Size', 'Ask_Size']

    Returns:
    - results: Dictionary with Kyle's Lambda and Amihud's Illiquidity Ratio.
    """
    # Merge trade data with liquidity data on the Time column
    data = pd.merge(trade_data, liquidity_data, on='Time')
    
    # Calculate Kyle's Lambda
    kyle_lambda = kyles_lambda(data)
    
    # Calculate Amihud's Illiquidity Ratio
    amihud_ratio = amihuds_illiquidity(data)
    
    results = {
        'Kyle_Lambda': kyle_lambda,
        'Amihud_Illiquidity_Ratio': amihud_ratio
    }
    
    return results

# Example usage
trade_data = pd.DataFrame({
    'Time': pd.date_range(start='2023-01-01', periods=100, freq='h'),
    'Trade_Price': np.random.uniform(100, 105, 100),
    'Trade_Volume': np.random.randint(1, 1000, 100)
})

liquidity_data = pd.DataFrame({
    'Time': pd.date_range(start='2023-01-01', periods=100, freq='h'),
    'Bid_Price': np.random.uniform(99, 104, 100),
    'Ask_Price': np.random.uniform(101, 106, 100),
    'Bid_Size': np.random.randint(100, 1000, 100),
    'Ask_Size': np.random.randint(100, 1000, 100)
})

results = market_impact_analysis(trade_data, liquidity_data)
print("Market Impact Analysis Results:")
print(results)