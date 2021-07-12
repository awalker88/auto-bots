import warnings
import datetime as dt
from functools import reduce
from typing import Union, List, Tuple

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tbats import BATS

from auto_bots.utils.error_metrics import mase, mse, rmse
from auto_bots.utils import validation as val
from auto_bots.utils.CandidateModel import CandidateModel


class AutoTS:
    """
    Automatic modeler that finds the best time-series method to model your data
    :param model_names: Models to consider when fitting. Currently supported models are
    'auto_arima', 'exponential_smoothing', 'tbats', and 'ensemble'. default is all available models
    :param error_metric: Which error metric to use when ranking models. Currently supported metrics
    are 'mase', 'mse', and 'rmse'. default='mase'
    :param seasonal_period: period of the data's seasonal trend. 3 would mean your data has quarterly
    trends. Supported models can use multiple seasonalities if a list is provided (Non-supported models
    will use the first item in list). None implies no seasonality.
    """

    def __init__(
        self,
        model_names: Union[Tuple[str], List[str]] = (
            "auto_arima",
            "exponential_smoothing",
            "tbats",
            "ensemble",
        ),
        error_metric: str = "mase",
        seasonal_period: Union[int, float, List[int], List[float]] = None,
        verbose: int = 0,
        auto_arima_args: dict = None,
        exponential_smoothing_args: dict = None,
        tbats_args: dict = None,
    ):
        self.verbose = verbose

        # fix mutable args
        if auto_arima_args is None:
            auto_arima_args = {}
        if exponential_smoothing_args is None:
            exponential_smoothing_args = {}
        if tbats_args is None:
            tbats_args = {}

        # input validation
        val.check_models(model_names)
        valid_error_metrics = ["mase", "mse", "rmse"]
        if error_metric.lower() not in valid_error_metrics:
            raise ValueError(f"Error metric must be one of {valid_error_metrics}")

        self.model_names = [model.lower() for model in model_names]
        self.error_metric = error_metric.lower()
        self.is_seasonal = True if seasonal_period is not None else False
        self.seasonal_period = val.set_seasonal_period(self, seasonal_period)
        self.auto_arima_args = auto_arima_args
        self.exponential_smoothing_args = exponential_smoothing_args
        self.tbats_args = tbats_args

        # Set during fitting or by other methods
        self.data = None
        self.series_column_name = None
        self.freq = None
        self.exogenous = None
        self.using_exogenous = False
        self.candidate_models = []
        self.fit_model = None
        self.fit_model_type = None
        self.best_model_error = None
        self.is_fitted = False
        self.prediction_index = None

        warnings.filterwarnings("ignore", module="statsmodels")

    def fit(
        self,
        data: pd.DataFrame,
        series_column_name: str,
        freq: str = "infer",
        exogenous: Union[str, list] = None,
    ) -> None:
        """
        Fit model to given training data.
        :param data: pandas dataframe containing series you would like to predict and any exogenous
        variables you'd like to be used. The dataframe's index MUST be a datetime index
        :param series_column_name: name of the column containing the series you would like to predict
        :param freq: frequency of your time series. pandas does a pretty good job inferring,
        but you can also specify via one of the options here
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        :param exogenous: column name or list of column names you would like to be used as exogenous
        regressors. auto_arima is the only model that supports exogenous regressors. The repressor
        columns should not be a constant or a trend
        """
        val.check_datetime_index(data)
        self.data = data
        self.series_column_name = series_column_name

        if freq == "infer":
            self.freq = pd.infer_freq(data.index.to_series())
        else:
            if freq not in list(pd.tseries.frequencies._offset_to_period_map):
                raise ValueError(
                    f"'{freq}' is not a recognized frequency option. "
                    f"`freq` must be 'infer' or one of the offsets described at this link: "
                    f"https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases "
                )

        # if user passes a string value (single column), make sure we can always assume exogenous is a list
        if isinstance(exogenous, str):
            exogenous = [exogenous]

        if exogenous is not None:
            self.using_exogenous = True
            self.exogenous = exogenous

        if "auto_arima" in self.model_names:
            self.candidate_models.append(self._fit_auto_arima())
            if self.verbose >= 1:
                print(
                    f"\tTrained auto_arima model with error {self.candidate_models[-1].error:.4f}"
                )
        if "exponential_smoothing" in self.model_names:
            self.candidate_models.append(self._fit_exponential_smoothing())
            if self.verbose >= 1:
                print(
                    f"\tTrained exponential_smoothing model with error {self.candidate_models[-1].error:.4f}"
                )
        if "tbats" in self.model_names:
            self.candidate_models.append(self._fit_tbats())
            if self.verbose >= 1:
                print(
                    f"\tTrained tbats model with error {self.candidate_models[-1].error:.4f}"
                )
        if "ensemble" in self.model_names:
            if self.candidate_models is None:
                raise ValueError("No candidate models to ensemble")
            self.candidate_models.append(self._fit_ensemble())
            if self.verbose >= 1:
                print(
                    f"\tTrained ensemble model with error {self.candidate_models[-1].error:.4f}"
                )

        self.candidate_models = sorted(self.candidate_models, key=lambda x: x.error)
        self.best_model_error = self.candidate_models[0].error
        self.fit_model = self.candidate_models[0].fit_model
        self.fit_model_type = self.candidate_models[0].model_type
        self.is_fitted = True

    def _fit_auto_arima(self) -> CandidateModel:
        """
        Fits an ARIMA model using pmdarima's auto_arima
        :return: Currently returns a list where the first item is the error on the test set, the
        second is the arima model, the third is the name of the model, and the fourth is the
        predictions made on the test set
        """
        exog = None
        if self.using_exogenous:
            exog = self.data[self.exogenous]

        auto_arima_seasonal_period = self.seasonal_period
        if self.seasonal_period is None:
            auto_arima_seasonal_period = (
                1  # need to use auto_arima default if there's no seasonality defined
            )
        else:
            # since auto_arima supports only 1 seasonality, select the first one as "main" seasonality
            auto_arima_seasonal_period = int(auto_arima_seasonal_period[0])

        try:
            model = auto_arima(
                self.data[self.series_column_name],
                error_action="ignore",
                supress_warning=True,
                seasonal=self.is_seasonal,
                m=auto_arima_seasonal_period,
                exogenous=exog,
                **self.auto_arima_args,
            )

        # occasionally while determining the necessary level of seasonal differencing, we get a weird
        # numpy dot product error due to array sizes mismatching. If that happens, we try using
        # Canova-Hansen test for seasonal differencing instead
        except ValueError:
            if self.verbose >= 2:
                print(
                    "\tSeasonal differencing for auto_arima failed. Trying Canova-Hansen method."
                )
            if (
                "seasonal_test" in self.auto_arima_args.keys()
                and self.auto_arima_args["seasonal_test"] == "ocsb"
            ):
                warnings.warn(
                    'Forcing `seasonal_test="ch"` as "ocsb" occasionally causes numpy errors',
                    UserWarning,
                )
            self.auto_arima_args["seasonal_test"] = "ch"
            model = auto_arima(
                self.data[self.series_column_name],
                error_action="ignore",
                supress_warning=True,
                seasonal=self.is_seasonal,
                m=auto_arima_seasonal_period,
                exogenous=exog,
                **self.auto_arima_args,
            )

        test_predictions = pd.DataFrame(
            {
                "actuals": self.data[self.series_column_name],
                "aa_test_predictions": model.predict_in_sample(exogenous=exog),
            }
        )

        test_error = self._error_metric(
            test_predictions, "aa_test_predictions", "actuals"
        )

        return CandidateModel(test_error, model, "auto_arima", test_predictions)

    def _fit_exponential_smoothing(self) -> CandidateModel:
        """
        Fits an exponential smoothing model using statsmodels's ExponentialSmoothing model
        :return: Currently returns a list where the first item is the error on the test set, the
        second is the exponential smoothing model, the third is the name of the model, and the
        fourth is the predictions made on the test set
        """
        # if user doesn't specify with kwargs, set these defaults
        if "trend" not in self.exponential_smoothing_args.keys():
            self.exponential_smoothing_args["trend"] = "add"
        if "seasonal" not in self.exponential_smoothing_args.keys():
            self.exponential_smoothing_args["seasonal"] = (
                "add" if self.seasonal_period is not None else None
            )

        es_seasonal_period = self.seasonal_period
        if self.seasonal_period is not None:
            es_seasonal_period = int(
                es_seasonal_period[0]
            )  # es supports only 1 seasonality

        model = ExponentialSmoothing(
            self.data[self.series_column_name],
            seasonal_periods=es_seasonal_period,
            **self.exponential_smoothing_args,
        ).fit()

        test_predictions = pd.DataFrame(
            {
                "actuals": self.data[self.series_column_name],
                "es_test_predictions": model.predict(
                    self.data.index[0], self.data.index[-1]
                ),
            }
        )

        error = self._error_metric(test_predictions, "es_test_predictions", "actuals")

        return CandidateModel(error, model, "exponential_smoothing", test_predictions)

    def _fit_tbats(self) -> CandidateModel:
        """
        Fits a BATS model using tbats's BATS model
        :return: Currently returns a list where the first item is the error on the test set, the
        second is the BATS model, the third is the name of the model, and the
        fourth is the predictions made on the test set
        """
        tbats_seasonal_periods = self.seasonal_period
        if self.seasonal_period is not None:
            tbats_seasonal_periods = self.seasonal_period

        # if user doesn't specify with kwargs, set these defaults
        if "n_jobs" not in self.tbats_args.keys():
            self.tbats_args["n_jobs"] = 1
        if "use_arma_errors" not in self.tbats_args.keys():
            self.tbats_args["use_arma_errors"] = False  # helps speed up modeling a bit

        model = BATS(
            seasonal_periods=tbats_seasonal_periods,
            use_box_cox=False,
            **self.tbats_args,
        )
        fit_model = model.fit(self.data[self.series_column_name])

        test_predictions = pd.DataFrame(
            {
                "actuals": self.data[self.series_column_name],
                "tb_test_predictions": fit_model.y_hat,
            }
        )
        error = self._error_metric(test_predictions, "tb_test_predictions", "actuals")

        return CandidateModel(error, fit_model, "tbats", test_predictions)

    def _fit_ensemble(self) -> CandidateModel:
        """
        Fits a model that is the ensemble of all other models specified during auto_bots's initialization
        :return: Currently returns a list where the first item is the error on the test set, the
        second is the exponential smoothing model, the third is the name of the model, and the
        fourth is the predictions made on the test set
        """
        model_predictions = [
            candidate.predictions for candidate in self.candidate_models
        ]
        all_predictions = reduce(
            lambda left, right: pd.merge(
                left,
                right.drop("actuals", axis="columns"),
                left_index=True,
                right_index=True,
            ),
            model_predictions,
        )
        predictions_columns = [
            col for col in all_predictions.columns if str(col).endswith("predictions")
        ]
        all_predictions["en_test_predictions"] = all_predictions[
            predictions_columns
        ].mean(axis="columns")

        error = self._error_metric(all_predictions, "en_test_predictions", "actuals")

        return CandidateModel(
            error, None, "ensemble", all_predictions[["actuals", "en_test_predictions"]]
        )

    def _error_metric(
        self, data: pd.DataFrame, predictions_column: str, actuals_column: str
    ) -> float:
        """
        Computes error using the error metric specified during initialization
        :param data: pandas dataframe containing predictions and actuals
        :param predictions_column: name of the predictions column
        :param actuals_column: name of the actuals column
        :return: error for given data
        """
        if self.error_metric == "mase":
            return mase(data, predictions_column, actuals_column)
        if self.error_metric == "mse":
            return mse(data, predictions_column, actuals_column)
        if self.error_metric == "rmse":
            return rmse(data, predictions_column, actuals_column)

    def _predict_auto_arima(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        last_data_date: dt.datetime,
        exogenous: pd.DataFrame = None,
    ) -> pd.Series:
        """Uses a fit ARIMA model to predict between the given dates"""
        # start date and end date are both in-sample
        if end_date <= self.data.index[-1]:
            preds = self.fit_model.predict_in_sample(
                start=self.data.index.get_loc(start_date),
                end=self.data.index.get_loc(end_date),
                exogenous=exogenous,
            )

        # start date is in-sample but end date is not
        elif start_date < self.data.index[-1] < end_date:
            num_extra_periods = (
                len(pd.date_range(start=last_data_date, end=end_date, freq=self.freq))
                - 1
            )

            in_sample_exog, out_of_sample_exog = None, None
            if self.using_exogenous:
                in_sample_exog = exogenous.iloc[
                    exogenous.index.get_loc(start_date) : exogenous.index.get_loc(
                        last_data_date
                    )
                    + 1
                ]
                out_of_sample_exog = exogenous.iloc[
                    exogenous.index.get_loc(last_data_date) + 1 :
                ]

            # get all in sample predictions and stitch them together with out of sample predictions
            in_sample_preds = self.fit_model.predict_in_sample(
                start=self.data.index.get_loc(start_date), exogenous=in_sample_exog
            )
            out_of_sample_preds = self.fit_model.predict(
                num_extra_periods, exogenous=out_of_sample_exog
            )
            preds = np.concatenate([in_sample_preds, out_of_sample_preds])

        # only possible scenario at this point is start date is 1 period past last data date
        else:
            periods_to_predict = len(
                pd.date_range(start=start_date, end=end_date, freq=self.freq)
            )
            preds = self.fit_model.predict(periods_to_predict, exogenous=exogenous)

        return pd.Series(
            preds, index=pd.date_range(start_date, end_date, freq=self.freq)
        )

    def _predict_exponential_smoothing(
        self, start_date: dt.datetime, end_date: dt.datetime
    ) -> pd.Series:
        """Uses a fit exponential smoothing model to predict between the given dates"""
        return self.fit_model.predict(start=start_date, end=end_date)

    def _predict_tbats(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        last_data_date: dt.datetime,
    ) -> pd.Series:
        """Uses a fit BATS model to predict between the given dates"""
        in_sample_preds = pd.Series(
            self.fit_model.y_hat,
            index=pd.date_range(
                start=self.data.index[0], end=self.data.index[-1], freq=self.freq
            ),
        )

        # start date and end date are both in-sample
        if end_date <= in_sample_preds.index[-1]:
            preds = in_sample_preds.loc[
                self.prediction_index[0] : self.prediction_index[-1]
            ]

        # start date is in-sample but end date is not
        elif start_date < self.data.index[-1] < end_date:
            num_extra_periods = (
                len(pd.date_range(start=last_data_date, end=end_date, freq=self.freq))
                - 1
            )
            # get all in sample predictions and stitch them together with out of sample predictions
            in_sample_portion = in_sample_preds.loc[start_date:]
            out_of_sample_portion = self.fit_model.forecast(num_extra_periods)
            preds = np.concatenate([in_sample_portion, out_of_sample_portion])

        # only possible scenario at this point is start date is 1 period past last data date
        else:
            preds = self.fit_model.forecast(len(self.prediction_index))

        return pd.Series(
            preds, index=pd.date_range(start=start_date, end=end_date, freq=self.freq)
        )

    def _predict_ensemble(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        last_data_date: dt.datetime,
        exogenous: pd.DataFrame,
    ) -> pd.Series:
        """Uses all other fit models to predict between the given dates and averages them"""
        ensemble_model_predictions = []

        if "auto_arima" in self.model_names:
            # todo: the way this works is kind of janky right now. probably want to move away from setting
            #  and resetting the fit_model attribute for each candidate model
            for candidate in self.candidate_models:
                if candidate.model_type == "auto_arima":
                    self.fit_model = candidate.fit_model
            preds = self._predict_auto_arima(
                start_date, end_date, last_data_date, exogenous
            )
            preds = preds.rename("auto_arima_predictions")
            ensemble_model_predictions.append(preds)

        if "exponential_smoothing" in self.model_names:
            for candidate in self.candidate_models:
                if candidate.model_type == "exponential_smoothing":
                    self.fit_model = candidate.fit_model
            preds = self._predict_exponential_smoothing(start_date, end_date)
            preds = preds.rename("exponential_smoothing_predictions")
            ensemble_model_predictions.append(preds)

        if "tbats" in self.model_names:
            for candidate in self.candidate_models:
                if candidate.model_type == "tbats":
                    self.fit_model = candidate.fit_model
            preds = self._predict_tbats(start_date, end_date, last_data_date)
            preds = preds.rename("tbats_predictions")
            ensemble_model_predictions.append(preds)

        all_predictions = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True
            ),
            ensemble_model_predictions,
        )
        all_predictions["en_test_predictions"] = all_predictions.mean(axis="columns")

        self.fit_model = None
        self.fit_model_type = "ensemble"

        return pd.Series(
            all_predictions["en_test_predictions"].values,
            index=pd.date_range(start=start_date, end=end_date, freq=self.freq),
        )

    def predict(
        self,
        start: Union[dt.datetime, str],
        end: Union[dt.datetime, str],
        exogenous: pd.DataFrame = None,
    ) -> pd.Series:
        """
        Generates predictions (forecasts) for dates between start and end (inclusive).
        :param start: date/time to begin forecast (inclusive), must be either within the date range
        given during fit or one interval after the last period given during fit
        :param end: date/time to end forecast (inclusive)
        :param exogenous: A dataframe of the exogenous regressor column(s) provided during fit().
        The dataframe should be of equal length to the number of predictions you would like to receive
        :return: A pandas Series of length equal to the number of intervals between star and
        end, where the interval is equal to the frequency given or inferred during fit.
        The series' will have a datetime index
        """
        if not self.is_fitted:
            raise AttributeError(
                "Model can't make predictions without first calling `.fit()`!"
            )

        val.validate_predict_dates(start, end)
        self._set_prediction_index(start, end)
        pred_start = self.prediction_index[0]
        pred_end = self.prediction_index[-1]

        last_period = self.data.index[-1]
        # check that start date is before or right after that last date given during training
        latest_valid_start = pd.date_range(
            self.data.index[-1], periods=2, freq=self.data.index.inferred_freq
        )[-1]
        if pred_start > latest_valid_start:
            raise ValueError(
                f"`start` must be no more than 1 period past the last date of data received"
                f" during fit. `start` = {pred_start} is too far ahead of latest valid start "
                f"{latest_valid_start}"
            )

        # check that start date comes after first date in training
        if pred_start < self.data.index[0]:
            raise ValueError(
                f"`start` must be later than the earliest date received during fit, "
                f"{self.data.index[0]}. Received `start` = {pred_start}"
            )

        # check that, if the user fit models with exogenous regressors, future values are provided
        # if we are predicting any out-of-sample periods
        if self.using_exogenous and last_period < pred_end and exogenous is None:
            raise ValueError(
                "Exogenous regressor(s) must be provided as a dataframe since they were provided during training"
            )

        if self.using_exogenous and not isinstance(exogenous.index, pd.DatetimeIndex):
            raise ValueError(
                "The index of your `exogenous` must be a series of datetimes"
            )

        # auto_arima requires a dataframe for the exogenous argument. If user provides a series, go
        # ahead and make it a dataframe, just to be nice :)
        if isinstance(exogenous, pd.Series):
            exogenous = pd.DataFrame(exogenous)

        # limit exogenous to dates that are specified by start and end date
        if self.using_exogenous:
            exogenous = exogenous[exogenous.index.isin(list(self.prediction_index))]

        # check that, if the user fit models with exogenous regressors, exogenous contains all necessary dates
        if (
            self.using_exogenous
            and self.prediction_index.tolist() != exogenous.index.tolist()
        ):
            raise ValueError(
                f"Exogenous regressor(s) must contain all dates in your prediction interval. The following dates are missing: {[d for d in self.prediction_index.tolist() if d not in exogenous.index.tolist()]}"
            )

        if self.fit_model_type == "auto_arima":
            return self._predict_auto_arima(
                pred_start, pred_end, last_period, exogenous
            )

        if self.fit_model_type == "exponential_smoothing":
            return self._predict_exponential_smoothing(pred_start, pred_end)

        if self.fit_model_type == "tbats":
            return self._predict_tbats(pred_start, pred_end, last_period)

        if self.fit_model_type == "ensemble":
            return self._predict_ensemble(pred_start, pred_end, last_period, exogenous)

    def _set_prediction_index(
        self, start: Union[dt.datetime, str], end: Union[dt.datetime, str]
    ):
        self.prediction_index = pd.date_range(start, end, freq=self.freq)
        if start != self.prediction_index[0]:
            warnings.warn(
                f"Given start {start} since is not valid with frequency {self.freq}. "
                f"Using {self.prediction_index[0]} instead",
                UserWarning,
            )
        if end != self.prediction_index[-1]:
            warnings.warn(
                f"Given end {end} since is not valid with frequency {self.freq}. "
                f"Using {self.prediction_index[-1]} instead",
                UserWarning,
            )
