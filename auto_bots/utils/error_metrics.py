import pandas as pd


def mse(data: pd.DataFrame, prediction: str, actuals: str):
    return ((data[prediction] - data[actuals]) ** 2).mean()


def rmse(data: pd.DataFrame, prediction: str, actuals: str):
    return mse(data, prediction, actuals)


def mape(data: pd.DataFrame, prediction: str, actuals: str):
    pass


def smape(data: pd.DataFrame, prediction: str, actuals: str):
    pass


def mase(data: pd.DataFrame, prediction: str, actuals: str, step_size: int = 1):
    data = data[[prediction, actuals]].copy()
    # add a small amount to avoid dividing by 0
    data['abs_shifted_error'] = abs(data[actuals] - data[actuals].shift(step_size)) + 0.01
    data['abs_prediction_error'] = abs(data[actuals] - data[prediction])

    avg_shifted_error = data[~data['abs_shifted_error'].isna()]['abs_shifted_error'].sum() / (len(data) - step_size)
    data['abs_scaled_error'] = data['abs_prediction_error'] / avg_shifted_error

    return data['abs_scaled_error'].mean()
