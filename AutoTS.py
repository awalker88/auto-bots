import warnings
from dateutil.relativedelta import relativedelta
import datetime as dt
from functools import reduce

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from error_metrics import mase
# from fbprophet import Prophet
# import matplotlib.pyplot as plot
from tbats import BATS

import validation as val


# todo: have user be able to access all models in addition to the one that works best
# todo: more basic models as options (ma, run_rate)
# todo: make example notebook
# todo: add dlm model
# todo: remove assumption that data is monthly
# todo: consider whether to have dynamic=True when predicting in sample for auto_arima
# todo: split up predict function to be like fit function
# todo: have seasonal_period support multiple periods when training tbats

class AutoTS:
    """
    Automatic modeler that finds the best time-series method to model your data
    :param model_names: Models to consider when fitting. Currently supported models are
    'auto_arima', 'exponential_smoothing', and 'tbats'. default=('auto_arima', 'exponential_smoothing')
    :param error_metric: Which error metric to use when ranking models. Currently supported metrics
    are 'mase', 'mse', and 'rmse'. default='mase'
    :param seasonal_period: period of the data's seasonal trend. 3 would mean your data has quarterly
    trends. None implies no seasonality. default=None
    :param holdout_period: number of periods to leave out as a test set when comparing candidate models.
    default=4
    """
    def __init__(self,
                 model_names=('auto_arima', 'exponential_smoothing', 'ensemble'),
                 # model_args: dict = None,
                 error_metric: str = 'mase',
                 seasonal_period: int = None,
                 # seasonality_mode: str = 'm',
                 holdout_period: int = 4,
                 verbose: bool = False
                 ):

        val.check_models(model_names)

        self.model_names = [model.lower() for model in model_names]
        # self.model_args = model_args
        self.error_metric = error_metric.lower()
        self.is_seasonal = True if seasonal_period is not None else False
        self.seasonal_period = seasonal_period
        self.holdout_period = holdout_period
        self.verbose = verbose

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
        self.is_fitted = False

        warnings.filterwarnings('ignore', module='statsmodels')

    def fit(self, data: pd.DataFrame, series_column_name: str, exogenous: list = None) -> None:
        """
        Fit model to given training data

        Note: exogenous variables are not yet supported
        :param data:
        :param series_column_name:
        :param exogenous:
        :return:
        """
        val.check_datetime_index(data)
        self._set_input_data(data, series_column_name)

        if exogenous is not None:
            self.using_exogenous = True
            self.exogenous = exogenous

        if 'auto_arima' in self.model_names:
            self.candidate_models.append(self._fit_auto_arima())
            if self.verbose:
                print(f'\tTrained auto_arima model with error {self.candidate_models[-1][0]}')
        if 'exponential_smoothing' in self.model_names:
            self.candidate_models.append(self._fit_exponential_smoothing(use_full_dataset=True))
            if self.verbose:
                print(f'\tTrained exponential_smoothing model with error {self.candidate_models[-1][0]}')
        if 'tbats' in self.model_names:
            self.candidate_models.append(self._fit_tbats(use_full_dataset=True))
            if self.verbose:
                print(f'\tTrained tbats model with error {self.candidate_models[-1][0]}')
        if 'ensemble' in self.model_names:
            if self.candidate_models is None:
                raise ValueError('No candidate models to ensemble')
            self.candidate_models.append(self._fit_ensemble())
            if self.verbose:
                print(f'\tTrained ensemble model with error {self.candidate_models[-1][0]}')

        # candidate_models[x][0] = model's error
        # candidate_models[x][1] = model object
        # candidate_models[x][2] = model's name
        # candidate_models[x][3] = model's predictions for the test set
        self.candidate_models = sorted(self.candidate_models, key=lambda x: x[0])
        self.best_model_error = self.candidate_models[0][0]
        self.fit_model = self.candidate_models[0][1]
        self.fit_model_type = self.candidate_models[0][2]
        self.is_fitted = True

        # now that we know the best model, retrain it on all data
        # if self.fit_model_type == 'auto_arima':
        #     self.fit_model = self._fit_auto_arima()[1]
        # elif self.fit_model_type == 'exponential_smoothing':
        #     self.fit_model = self._fit_exponential_smoothing(use_full_dataset=True)[1]
        # elif self.fit_model_type == 'tbats':
        #     self.fit_model = self._fit_tbats(use_full_dataset=True)[1]
        # elif self.fit_model_type == 'ensemble':
        #     self.fit_model = self._fit_ensemble()

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

        test_predictions = pd.DataFrame({'actuals': self.testing_data[self.series_column_name],
                                         'aa_test_predictions': model.predict(n_periods=len(self.testing_data),
                                                                              exogenous=test_exog)})

        test_error = self._error_metric(test_predictions, 'aa_test_predictions', 'actuals')

        # now that we have train score, we'll want the fitted model to have all available data if it's chosen
        model.update(self.testing_data[self.series_column_name], exogenous=test_exog)

        return [test_error, model, 'auto_arima', test_predictions]

    def _fit_exponential_smoothing(self, use_full_dataset: bool = False):
        if use_full_dataset:
            model_data = self.data
        else:
            model_data = self.training_data

        model = ExponentialSmoothing(model_data[self.series_column_name],
                                     seasonal_periods=self.seasonal_period,
                                     trend='add',
                                     seasonal='add',
                                     use_boxcox=False,
                                     initialization_method='estimated').fit()

        test_predictions = pd.DataFrame(
            {'actuals': self.testing_data[self.series_column_name],
             'es_test_predictions': model.predict(
                 len(self.training_data), len(self.training_data) + self.holdout_period - 2
             )})

        error = self._error_metric(test_predictions, 'es_test_predictions', 'actuals')

        return [error, model, 'exponential_smoothing', test_predictions]

    def _fit_tbats(self, use_full_dataset: bool = False, use_simple_model: bool = True):
        if use_full_dataset:
            model_data = self.data
        else:
            model_data = self.training_data

        # turning off arma and box_cox cuts training time by about a third
        if use_simple_model:
            model = BATS(
                seasonal_periods=[self.seasonal_period] + [2, 4],
                use_arma_errors=False,
                use_box_cox=False
            )
        else:
            model = BATS(
                seasonal_periods=[self.seasonal_period] + [2, 4],
            )

        fitted_model = model.fit(model_data[self.series_column_name])
        test_predictions = pd.DataFrame({'actuals': self.testing_data[self.series_column_name],
                                         'tb_test_predictions': fitted_model.forecast(len(self.testing_data))})
        error = self._error_metric(test_predictions, 'tb_test_predictions', 'actuals')

        return [error, fitted_model, 'tbats', test_predictions]

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
        model_predictions = [candidate[3] for candidate in self.candidate_models]
        all_predictions = reduce(lambda left, right: pd.merge(left, right.drop('actuals', axis='columns'),
                                                              left_index=True, right_index=True),
                                 model_predictions)
        predictions_columns = [col for col in all_predictions.columns if str(col).endswith('predictions')]
        all_predictions['en_test_predictions'] = all_predictions[predictions_columns].mean(axis='columns')

        error = self._error_metric(all_predictions, 'en_test_predictions', 'actuals')

        return [error, None, 'ensemble', all_predictions[['actuals', 'en_test_predictions']]]

    def _error_metric(self, data: pd.DataFrame, predictions_column: str, actuals_column: str):
        if self.error_metric == 'mase':
            return mase(data, predictions_column, actuals_column)

    def _set_input_data(self, data: pd.DataFrame, series_column_name: str):
        self.data = data
        self.training_data = data.iloc[:-self.holdout_period, :]
        self.testing_data = data.iloc[-self.holdout_period:, :]
        self.series_column_name = series_column_name

    def _predict_auto_arima(self, start_date: dt.datetime, end_date: dt.datetime, last_data_date: dt.datetime) -> pd.Series:
        # start date and end date are both in-sample
        if start_date < self.data.index[-1] and end_date <= self.data.index[-1]:
            preds = self.fit_model.predict_in_sample(start=self.data.index.get_loc(start_date),
                                                     end=self.data.index.get_loc(end_date))

        # start date is in-sample but end date is not
        elif start_date < self.data.index[-1] < end_date:
            num_extra_months = (end_date.year - last_data_date.year) * 12 + (end_date.month - last_data_date.month)
            # get all in sample predictions and stitch them together with out of sample predictions
            in_sample_preds = self.fit_model.predict_in_sample(start=self.data.index.get_loc(start_date))
            out_of_sample_preds = self.fit_model.predict(num_extra_months)
            preds = np.concatenate([in_sample_preds, out_of_sample_preds])

        # only possible scenario at this point is start date is 1 month past last data date
        else:
            months_to_predict = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
            preds = self.fit_model.predict(months_to_predict)

        return pd.Series(preds, index=pd.date_range(start_date, end_date, freq='MS'))

    def _predict_exponential_smoothing(self, start_date: dt.datetime, end_date: dt.datetime):
        return self.fit_model.predict(start=start_date, end=end_date)

    def _predict_tbats(self, start_date: dt.datetime, end_date: dt.datetime, last_data_date: dt.datetime):
        in_sample_preds = pd.Series(self.fit_model.y_hat,
                                    index=pd.date_range(start=self.data.index[0],
                                                        end=self.data.index[-1], freq='MS'))

        # start date and end date are both in-sample
        if start_date < in_sample_preds.index[-1] and end_date <= in_sample_preds.index[-1]:
            preds = in_sample_preds.loc[start_date:end_date]

        # start date is in-sample but end date is not
        elif start_date < self.data.index[-1] < end_date:
            num_extra_months = (end_date.year - last_data_date.year) * 12 + (end_date.month - last_data_date.month)
            # get all in sample predictions and stitch them together with out of sample predictions
            in_sample_portion = in_sample_preds.loc[start_date:]
            out_of_sample_portion = self.fit_model.forecast(num_extra_months)
            preds = np.concatenate([in_sample_portion, out_of_sample_portion])

        # only possible scenario at this point is start date is 1 month past last data date
        else:
            months_to_predict = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
            preds = self.fit_model.forecast(months_to_predict)

        return pd.Series(preds, index=pd.date_range(start=start_date, end=end_date, freq='MS'))

    def _predict_ensemble(self, start_date: dt.datetime, end_date: dt.datetime, last_data_date: dt.datetime):
        ensemble_model_predictions = []

        if 'auto_arima' in self.model_names:
            # todo: the way this works is kind of janky right now. probably want to move away from setting
            # and resetting the fit_model attribute for each candidate model
            for model in self.candidate_models:
                if model[2] == 'auto_arima':
                    self.fit_model = model[1]
            preds = self._predict_auto_arima(start_date, end_date, last_data_date)
            preds = preds.rename('auto_arima_predictions')
            ensemble_model_predictions.append(preds)

        if 'exponential_smoothing' in self.model_names:
            for model in self.candidate_models:
                if model[2] == 'exponential_smoothing':
                    self.fit_model = model[1]
            preds = self._predict_exponential_smoothing(start_date, end_date)
            preds = preds.rename('exponential_smoothing_predictions')
            ensemble_model_predictions.append(preds)

        if 'tbats' in self.model_names:
            for model in self.candidate_models:
                if model[2] == 'tbats':
                    self.fit_model = model[1]
            preds = self._predict_tbats(start_date, end_date, last_data_date)
            preds = preds.rename('tbats_predictions')
            ensemble_model_predictions.append(preds)

        all_predictions = reduce(lambda left, right: pd.merge(left, right,
                                                              left_index=True, right_index=True),
                                 ensemble_model_predictions)
        all_predictions['en_test_predictions'] = all_predictions.mean(axis='columns')

        self.fit_model = None

        return pd.Series(all_predictions['en_test_predictions'].values,
                         index=pd.date_range(start=start_date, end=end_date, freq='MS'))

    def predict(self, start_date: dt.datetime, end_date: dt.datetime) -> pd.Series:
        """
        Generates predictions (forecasts) for dates between start_date and end_date (inclusive).
        :param start_date: date to begin forecast (inclusive), must be either within the date range
        given during fit or the month immediately following the last date given during fit
        :param end_date: date to end forecast (inclusive)
        :return: A pandas Series of length equal to the number of months between start_date and
        end_date. The series' will have a datetime index
        """
        ### checks on data
        if not self.is_fitted:
            raise AttributeError('Model must be fitted to be able to make predictions. Use the '
                                 '`fit` method to fit before predicting')

        # check inputs are datetimes
        if not (isinstance(start_date, dt.datetime) and isinstance(end_date, dt.datetime)):
            raise TypeError('Both `start_date` and `end_date` must be datetime objects')

        # check start date comes before end date
        if start_date + relativedelta(months=+1) > end_date:
            raise ValueError('`start_date` must be at least one month before `end_date`')

        # check that start date is before or right after that last date given during training
        last_data_date = self.data.index[-1]
        if start_date > (last_data_date + relativedelta(months=+1)):
            raise ValueError(f'`start_date` must be no more than 1 month past the last date of data received'
                             f' during fit". Received `start date` is '
                             f'{(start_date.year - last_data_date.year) * 12 + (start_date.month - last_data_date.month)}'
                             f'months after last date in data {last_data_date}')

        # check that start date comes after first date in training
        if start_date < self.data.index[0]:
            raise ValueError(f'`start_date` must be later than the earliest date received during fit')

        ### auto arima
        if self.fit_model_type == 'auto_arima':
            return self._predict_auto_arima(start_date, end_date, last_data_date)

        ### exponential smoothing
        if self.fit_model_type == 'exponential_smoothing':
            return self._predict_exponential_smoothing(start_date, end_date)

        ### tbats
        if self.fit_model_type == 'tbats':
            return self._predict_tbats(start_date, end_date, last_data_date)

        if self.fit_model_type == 'ensemble':
            return self._predict_ensemble(start_date, end_date, last_data_date)




