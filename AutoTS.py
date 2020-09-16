import numpy as np
from pmdarima import auto_arima


class AutoTS:
    def __init__(self,
                 models=('auto_armia', 'exponential_smoothing', 'TBATS', 'prophet', 'ensemble'),
                 model_args: dict = None,
                 error_metric: str = 'mase',
                 seasonal: bool = True
                 ):
        if type(models) not in [tuple, list]:
            raise TypeError('`models` argument must a list or tuple')

        self.models = [model.lower() for model in models]
        self.model_args = model_args
        self.error_metric = error_metric.lower()
        self.seasonal = seasonal
        self.x = None
        self.y = None
        self.using_exogenous = False
        self.exogenous_data = None

    def fit(self, x, y, exogenous=None):
        self.x = x
        self.y = y
        if exogenous is not None:
            self.using_exogenous = True
        arima = self._fit_auto_arima()
        pass

    def _fit_auto_arima(self):
        arima = auto_arima(self.x, exogenous=self.exogenous_data, error_action='ignore',
                           supress_warning=True, seasonal=self.seasonal)
        arima =

        return arima
