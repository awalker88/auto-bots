import pandas as pd
from pmdarima import auto_arima
from numpy import inf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from error_metrics import mase

import warnings


class AutoTS:
    def __init__(self,
                 models=('auto_armia', 'exponential_smoothing', 'TBATS', 'prophet', 'ensemble'),
                 model_args: dict = None,
                 # error_metric: str = 'mase',
                 is_seasonal: bool = True,
                 seasonal_period: int = 4,
                 holdout_length: int = 4
                 ):

        if type(models) not in [tuple, list]:
            raise TypeError('`models` argument must a list or tuple')

        self.models = [model.lower() for model in models]
        self.model_args = model_args
        # self.error_metric = error_metric.lower()
        self.is_seasonal = is_seasonal
        self.seasonal_period = seasonal_period
        self.holdout_length = holdout_length

        self.data = None
        self.x = None
        self.exogenous = None
        self.using_exogenous = False

        warnings.filterwarnings('ignore', module='statsmodels')

    def fit(self, data: pd.DataFrame, x: str, exogenous: list = None):
        self.data = data
        self.x = x

        if exogenous is not None:
            self.using_exogenous = True
            self.exogenous = exogenous

        arima = self._fit_auto_arima()
        es = self._fit_exponential_smoothing()

    def _fit_auto_arima(self):
        exog = None
        if self.using_exogenous:
            exog = self.data[self.exogenous]
        arima = auto_arima(self.data[self.x], exogenous=exog, error_action='ignore',
                           supress_warning=True, seasonal=self.is_seasonal, m=self.seasonal_period)
        return arima

    def _fit_exponential_smoothing(self, hypertune: bool = True):
        params = {
            'trend': ['add', 'mul'],
            'is_seasonal': ['add', 'mul'],
        }
        if hypertune:
            temp_df = self.data.copy()
            lowest_error = inf
            best_model = None
            for trend in params['trend']:
                for seasonal in params['is_seasonal']:
                    es_model = ExponentialSmoothing(temp_df[self.x],
                                                    seasonal_periods=self.seasonal_period,
                                                    trend=trend,
                                                    seasonal=seasonal,
                                                    use_boxcox=True,
                                                    initialization_method="estimated").fit()
                    predictions = es_model.predict(start=0, end=len(temp_df) + 1)
                    temp_df['prediction'] = predictions
                    error = mase(temp_df, 'prediction', 'expense_plan_amount')
                    if error < lowest_error:
                        lowest_error = error
                        best_model = es_model
            return best_model

        es_model = ExponentialSmoothing(self.data[self.x],
                                        seasonal_periods=self.seasonal_period,
                                        trend='add',
                                        seasonal='add',
                                        use_boxcox=True,
                                        initialization_method="estimated").fit()

        return es_model

    def _fit_tbats(self):
        pass

    def _fit_prophet(self):
        pass

    def _fit_ensemble(self):
        pass

    def _error_metric(self):
        pass

    def predict(self):
        pass

