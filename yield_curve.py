import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Nelson-Siegel model function
def nelson_siegel(t, beta0, beta1, beta2, tau):
    return beta0 + beta1 * ((1 - np.exp(-t/tau)) / (t/tau)) + beta2 * (((1 - np.exp(-t/tau)) / (t/tau)) - np.exp(-t/tau))

def fit_nelson_siegel(yields, maturities):
    # Fit the Nelson-Siegel model to the historical yield data
    params = []
    for i in range(yields.shape[0]):
        popt, _ = curve_fit(nelson_siegel, maturities, yields.iloc[i, :], maxfev=10000)
        params.append(popt)
    return np.array(params)

def forecast_parameters(params, prediction_horizon):
    """
    Forecast Nelson-Siegel parameters using ARIMA models.
    
    Parameters:
    - params: Array of historical Nelson-Siegel parameters.
    - prediction_horizon: Number of periods into the future to predict.
    
    Returns:
    - forecasted_params: Forecasted parameters for the specified prediction horizon.
    """
    forecasted_params = np.zeros((prediction_horizon, params.shape[1]))
    for i in range(params.shape[1]):
        model = ARIMA(params[:, i], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=prediction_horizon)
        forecasted_params[:, i] = forecast
    return forecasted_params

def predict_yield_curve_ns(historical_yields, maturities, prediction_horizon):
    """
    Predict future yield curves using the Nelson-Siegel model.
    
    Parameters:
    - historical_yields: DataFrame with historical yields for different maturities.
    - maturities: List of maturities corresponding to the yields.
    - prediction_horizon: The number of periods into the future to predict.

    Returns:
    - predicted_yields: DataFrame with predicted yields for different maturities.
    """
    # Fit the Nelson-Siegel model to the historical yield data
    params = fit_nelson_siegel(historical_yields, maturities)
    
    # Forecast future parameters using ARIMA
    forecasted_params = forecast_parameters(params, prediction_horizon)
    
    # Predict future yields using the forecasted parameters
    predicted_yields = pd.DataFrame(index=range(prediction_horizon), columns=historical_yields.columns)
    
    for t in range(prediction_horizon):
        predicted_yields.iloc[t, :] = nelson_siegel(maturities, *forecasted_params[t, :])
    
    return predicted_yields

# Example usage
historical_yield_data = pd.DataFrame({
    '2Y': [0.01, 0.015, 0.02, 0.025, 0.03],
    '5Y': [0.015, 0.02, 0.025, 0.03, 0.035],
    '10Y': [0.02, 0.025, 0.03, 0.035, 0.04],
    '20Y': [0.025, 0.03, 0.035, 0.04, 0.045],
    '30Y': [0.03, 0.035, 0.04, 0.045, 0.05]
})

maturities = np.array([2, 5, 10, 20, 30])  # Maturities in years
prediction_horizon = 5  # Predict the next 5 periods

predicted_yields = predict_yield_curve_ns(historical_yield_data, maturities, prediction_horizon)

print("Predicted Yields:")
print(predicted_yields)