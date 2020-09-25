import warnings
from math import floor
from dateutil.relativedelta import relativedelta
import datetime as dt

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from numpy import inf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from error_metrics import mase
# from fbprophet import Prophet
import matplotlib.pyplot as plot
from tbats import TBATS, BATS

from validate_inputs import validate_inputs


# todo: have user be able to access all models in addition to the one that works best
# todo: more basic models as options (ma, run_rate)
# todo: make example notebook
# todo: add dlm model
# todo: have flexible options for predict (int for future only, start_date/end_date)
# todo: have a fit model class that contains info about a fit model like it's name and error
# todo: remove assumption that data is monthly
# todo: consider whether to have dynamic=True when predicting in sample for auto_arima

class AutoTS:
    def __init__(self,
                 model_names=('auto_arima', 'exponential_smoothing', 'tbats', 'prophet', 'ensemble'),
                 model_args: dict = None,
                 error_metric: str = 'mase',
                 is_seasonal: bool = True,
                 seasonal_period: int = 3,
                 seasonality_mode: str = 'm',
                 holdout_period: int = 4
                 ):

        validate_inputs(model_names)

        self.model_names = [model.lower() for model in model_names]
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
        self.candidate_models = []
        self.fit_model = None
        self.fit_model_type = None
        self.best_model_error = None

        warnings.filterwarnings('ignore', module='statsmodels')

    def fit(self, data: pd.DataFrame, series_column_name: str, exogenous: list = None):
        self._set_input_data(data, series_column_name)

        if exogenous is not None:
            self.using_exogenous = True
            self.exogenous = exogenous

        if 'auto_arima' in self.model_names:
            self.candidate_models.append(self._fit_auto_arima())
        if 'exponential_smoothing' in self.model_names:
            self.candidate_models.append(self._fit_exponential_smoothing())
        if 'tbats' in self.model_names:
            self.candidate_models.append(self._fit_tbats())

        self.candidate_models = sorted(self.candidate_models, key=lambda x: x[0])
        self.best_model_error = self.candidate_models[0][0]
        self.fit_model = self.candidate_models[0][1]
        self.fit_model_type = self.candidate_models[0][2]

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

        error = self._error_metric(predictions, 'predictions', 'actuals')

        # now that we have train score, we'll want the fitted model to have all available data if it's chosen
        model.update(self.testing_data[self.series_column_name], exogenous=test_exog)

        return [error, model, 'auto_arima']

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

        predictions = pd.DataFrame({'actuals': self.testing_data[self.series_column_name],
                                    'predictions': best_model.predict(len(self.training_data),
                                                                      len(self.training_data) + self.holdout_period - 2)})
        error = self._error_metric(predictions, 'predictions', 'actuals')

        return [error, best_model, 'exponential_smoothing']

    def _fit_tbats(self):
        model = BATS(
            seasonal_periods=[3],
            use_arma_errors=False,
            use_box_cox=False
        )
        fitted_model = model.fit(self.training_data[self.series_column_name])
        predictions = pd.DataFrame({'actuals': self.testing_data[self.series_column_name],
                                    'predictions': fitted_model.forecast(len(self.testing_data))})
        error = self._error_metric(predictions, 'predictions', 'actuals')

        return [error, fitted_model, 'tbats']

    # def _fit_prophet(self):
    #
    #     # have to check if the dataframe already contains
    #     datetime_index_name = self.data.index.name
    #     if datetime_index_name in self.data.columns:
    #         proph_df = self.data.reset_index(drop=True)
    #     else:
    #         proph_df = self.data.reset_index()
    #
    #     # prophet requires the input dataframe to have two columns named 'ds' and 'y'
    #     proph_df = proph_df[[datetime_index_name, self.series_column_name]].\
    #         rename({datetime_index_name: 'ds', self.series_column_name: 'y'}, axis='columns')
    #
    #     model = Prophet(changepoint_range=1.,  # set to 1 since we're fitting only on training data
    #                     )
    #
    #     return Prophet().fit(proph_df)

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

    def predict(self, start_date: dt.datetime, end_date: dt.datetime):
        # check inputs are datetimes
        if not (isinstance(start_date, dt.datetime) and isinstance(end_date, dt.datetime)):
            raise TypeError('Both `start_date` and `end_date` must be datetime objects')

        # check start date comes before end date
        if start_date >= (end_date + relativedelta(months=-1)):
            raise ValueError('`start_date` must be at least one month before `end_date`')

        # check that start date is before or right after that last date given during training
        last_data_date = self.data.index[-1]
        if start_date > (last_data_date + relativedelta(months=+1)):
            raise ValueError(f'`start_date` must be no more than 1 month past the last date of data received'
                             f' during fit". `start date` is currently '
                             f'{(start_date.year - last_data_date.year) * 12 + (start_date.month - last_data_date.month)}'
                             f'months after last date in data {last_data_date}')

        if self.fit_model_type == 'auto_arima':
            # start date and end date are both in-sample
            if start_date < self.data.index[-1] and end_date <= self.data.index[-1]:
                preds = self.fit_model.predict_in_sample(start=self.data.index.get_loc(start_date),
                                                         end=self.data.index.get_loc(start_date))

            # start date is in-sample but end date is not
            elif start_date < self.data.index[-1] < end_date:
                extra_months = (end_date.year - last_data_date.year) * 12 + (end_date.month - last_data_date.month)
                # get all in sample predictions and stitch them together with out of sample predictions
                in_sample_preds = self.fit_model.predict_in_sample(start=self.data.index.get_loc(start_date))
                out_of_sample_preds = self.fit_model.predict(extra_months)
                preds = np.concatenate([in_sample_preds, out_of_sample_preds])

            # only possible scenario at this point is start date is 1 month past last data date
            months_to_predict = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
            preds = self.fit_model.predict(months_to_predict)
            return pd.Series(preds, index=pd.date_range(start_date, end_date, freq='MS'))

        elif self.fit_model_type == 'exponential_smoothing':
            return self.fit_model.predict(start=start_date, end=end_date)

        return -1


