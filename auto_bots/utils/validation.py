from typing import Union, List
import datetime as dt
import warnings

import pandas as pd


def check_models(models):
    if type(models) not in [tuple, list]:
        raise TypeError("`models` argument must a list or tuple")

    valid_models = ["auto_arima", "exponential_smoothing", "tbats", "ensemble"]
    if len(models) == 0:
        raise ValueError(f"`models` argument must contain at least one of {valid_models}")

    invalid_models = [model for model in models if model not in valid_models]
    if len(invalid_models) > 0:
        raise ValueError(f"The following models are not supported: {invalid_models}")

    if len(models) <= 2 and "ensemble" in models:
        raise ValueError(
            "If you wish to have `ensemble` be a candidate model, you must specify at "
            "least two additional valid models"
        )


def check_datetime_index(series_df: pd.DataFrame):
    if not isinstance(series_df.index, pd.DatetimeIndex):
        raise TypeError("The index of your dataframe must be a series of datetimes")


def validate_predict_dates(start_date: Union[dt.datetime, str], end_date: Union[dt.datetime, str]):
    # check inputs are datetimes or strings that are capable of being turned into datetimes
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    elif not isinstance(start_date, dt.datetime):
        raise TypeError("`start_date` must be a str or datetime-like object")
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    elif not isinstance(end_date, dt.datetime):
        raise TypeError("`end_date` must be a str or datetime-like object")

    # check start date doesn't come before end_date
    if start_date > end_date:
        raise ValueError("`end_date` must come after `start_date`")


def set_seasonal_period(self, given_period: Union[int, float, List[int], List[float]]):
    """Checks we're given proper seasonal period(s) and ensures we can assume self.seasonal_period will be a list"""
    if given_period is None:
        return None

    if isinstance(given_period, (int, float)):
        if isinstance(given_period, float) and self.model_names != ["tbats"] and self.verbose >= 2:
            warnings.warn("Float seasonal periods are only supported in `tbats`, other models will round down to int")

        return [given_period]

    if isinstance(given_period, list):
        for item in given_period:
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"Seasonal periods must be of type `int` or `float`, received type `{type(item)}` for {item}"
                )
        return given_period
