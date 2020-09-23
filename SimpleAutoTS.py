import warnings

import pandas as pd
from pmdarima import auto_arima


class SimpleAutoTS:
    def __init__(self,
                 models=('auto_arima', 'exponential_smoothing', 'tbats', 'prophet', 'ensemble'),
                 model_args: dict = None,
                 error_metric: str = 'mase',
                 is_seasonal: bool = True,
                 seasonal_period: int = 3,
                 seasonality_mode: str = 'm',
                 holdout_period: int = 4
                 ):

        self.models = [model.lower() for model in models]
        self.model_args = model_args
        self.error_metric = error_metric.lower()
        self.is_seasonal = is_seasonal
        self.seasonal_period = seasonal_period
        self.holdout_period = holdout_period

        # Set during fitting or by other methods
        self.data = None
        self.training_data = None
        self.testing_data = None
        self.series_column_name = None
        self.exogenous = None
        self.using_exogenous = False
        self.model = None

        warnings.filterwarnings('ignore', module='statsmodels')

    def fit(self, data: pd.DataFrame, series_column_name: str, exogenous: list = None):
        self._set_input_data(data, series_column_name)

        if exogenous is not None:
            self.using_exogenous = True
            self.exogenous = exogenous

        arima, arima_error = self._fit_auto_arima()
        self.model = arima

    def _fit_auto_arima(self):
        train_exog = None
        test_exog = None
        if self.using_exogenous:
            train_exog = self.training_data[self.exogenous]
            test_exog = self.testing_data[self.exogenous]

        model = auto_arima(self.training_data[self.series_column_name], exogenous=train_exog,
                           error_action='ignore',
                           supress_warning=True,
                           # trace=True,
                           seasonal=self.is_seasonal, m=self.seasonal_period
                           )

        predictions = pd.DataFrame({'actuals': self.testing_data[self.series_column_name],
                                    'predictions': model.predict(n_periods=len(self.testing_data), exogenous=test_exog)})

        error = self.mase(predictions, 'predictions', 'actuals')

        return model, error

    def _set_input_data(self, data: pd.DataFrame, series_column_name: str):
        self.data = data
        self.training_data = data.iloc[:-self.holdout_period, :]
        self.testing_data = data.iloc[-self.holdout_period:, :]
        self.series_column_name = series_column_name

    def predict(self, n_periods):
        return self.model.predict(n_periods=n_periods)

    def mase(self, data: pd.DataFrame, prediction: str, actuals: str, step_size: int = 1):
        data = data[[prediction, actuals]].copy()
        data['abs_shifted_error'] = abs(data[actuals] - data[actuals].shift(step_size))
        data['abs_prediction_error'] = abs(data[actuals] - data[prediction])

        avg_shifted_error = data[~data['abs_shifted_error'].isna()]['abs_shifted_error'].sum() / (len(data) - step_size)
        data['abs_scaled_error'] = data['abs_prediction_error'] / avg_shifted_error

        return data['abs_scaled_error'].mean()

