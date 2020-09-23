import warnings
from math import floor

import pandas as pd
from pmdarima import auto_arima
from numpy import inf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from error_metrics import mase
from fbprophet import Prophet

from validate_inputs import validate_inputs


# todo: have user be able to access all models in addition to the one that works best
# todo: more basic models as options (ma, run_rate)
# todo: make example notebook
# todo: add dlm model

class AutoTS:
    def __init__(self,
                 models=('auto_arima', 'exponential_smoothing', 'tbats', 'prophet', 'ensemble'),
                 model_args: dict = None,
                 error_metric: str = 'mase',
                 is_seasonal: bool = True,
                 seasonal_period: int = 3,
                 seasonality_mode: str = 'm',
                 holdout_period: int = 3
                 ):

        validate_inputs(models)

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

        warnings.filterwarnings('ignore', module='statsmodels')

    def fit(self, data: pd.DataFrame, series_column_name: str, exogenous: list = None):
        self._set_input_data(data, series_column_name)

        if exogenous is not None:
            self.using_exogenous = True
            self.exogenous = exogenous

        arima, arima_error = self._fit_auto_arima()
        es, es_error = self._fit_exponential_smoothing()
        print(arima_error, es_error)

        return None

    def _fit_auto_arima(self):
        train_exog = None
        test_exog = None
        if self.using_exogenous:
            train_exog = self.training_data[self.exogenous]
            test_exog = self.testing_data[self.exogenous]

        model = auto_arima(self.training_data[self.series_column_name], exogenous=train_exog,
                           error_action='ignore',
                           supress_warning=True,
                           seasonal=self.is_seasonal, m=self.seasonal_period
                           )

        predictions = pd.DataFrame({'actuals': self.testing_data[self.series_column_name],
                                    'predictions': model.predict(n_periods=len(self.testing_data), exogenous=test_exog)})

        error = self._error_metric(predictions, 'predictions', 'actuals')

        return model, error

    def _fit_exponential_smoothing(self, hypertune: bool = False):
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
                    es_model = ExponentialSmoothing(temp_df[self.series_column_name],
                                                    seasonal_periods=self.seasonal_period,
                                                    trend=trend,
                                                    seasonal=seasonal,
                                                    use_boxcox=False,
                                                    initialization_method='estimated').fit()
                    predictions = es_model.predict(start=0, end=len(temp_df) + 1)
                    temp_df['prediction'] = predictions
                    error = self._error_metric(temp_df, 'prediction', 'expense_plan_amount')
                    if error < lowest_error:
                        lowest_error = error
                        best_model = es_model
            return best_model

        best_model = ExponentialSmoothing(self.data[self.series_column_name],
                                          seasonal_periods=self.seasonal_period,
                                          trend='add',
                                          seasonal='add',
                                          use_boxcox=False,
                                          initialization_method='estimated').fit()

        predictions = pd.DataFrame({'actuals': self.data[self.series_column_name],
                                    'predictions': best_model.predict(0)})
        error = self._error_metric(predictions, 'predictions', 'actuals')

        return best_model, error

    def _fit_tbats(self):
        pass

    def _fit_prophet(self):

        # have to check if the dataframe already contains
        datetime_index_name = self.data.index.name
        if datetime_index_name in self.data.columns:
            proph_df = self.data.reset_index(drop=True)
        else:
            proph_df = self.data.reset_index()

        # prophet requires the input dataframe to have two columns named 'ds' and 'y'
        proph_df = proph_df[[datetime_index_name, self.series_column_name]].\
            rename({datetime_index_name: 'ds', self.series_column_name: 'y'}, axis='columns')

        model = Prophet(changepoint_range=1.,  # set to 1 since we're fitting only on training data
                        )

        return Prophet().fit(proph_df)

    def _fit_ensemble(self):
        pass

    def _error_metric(self, data: pd.DataFrame, predictions_column: str, actuals_column: str):
        if self.error_metric == 'mase':
            return mase(data, predictions_column, actuals_column)

    def _set_input_data(self, data: pd.DataFrame, series_column_name: str):
        self.data = data
        self.training_data = data.iloc[:-self.holdout_period, :]
        self.testing_data = data.iloc[-self.holdout_period:, :]
        self.series_column_name = series_column_name

    def predict(self):
        pass

