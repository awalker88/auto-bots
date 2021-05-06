from typing import Union, List, Tuple
from dateutil.relativedelta import relativedelta
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt

from AutoTS.AutoTS import AutoTS
from AutoTS.utils.error_metrics import mase


def seasonality_tests(data: pd.DataFrame):
    possible_seasonalities = [3, [3], [12, 3]]
    for seasonality in possible_seasonalities:
        print(f"Starting seasonality test with seasonality {seasonality}")
        model = AutoTS(seasonal_period=seasonality)
        model.fit(data=data, series_column_name="ts")
        print(f"\tTest passed")


def test_accuracy(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    dataset_name: str,
    seasonality: Union[int, List[int]] = None,
):
    print(f"Starting accuracy test for dataset {dataset_name}")
    model = AutoTS(seasonal_period=seasonality, verbose=2)
    model.fit(data=train_data, series_column_name="ts")

    forecast = pd.Series(
        data=model.predict(start=test_data.index[0], end=test_data.index[-1]),
        index=test_data.index,
        name="forecast",
    )
    forecast = pd.merge(forecast, test_data, right_index=True, left_index=True)
    forecast_error = mase(forecast, "forecast", "ts")

    if dataset_name == "airline":
        assert model.best_model_error < 0.4
        assert forecast_error < 0.8
        print(
            f"Passed airline accuracy test with train error = {model.best_model_error:.3f} and "
            f"test error {forecast_error:.3f}"
        )
    elif dataset_name == "shampoo":
        assert model.best_model_error < 0.6
        assert forecast_error < 1.
        print(
            f"Passed shampoo accuracy test with train error = {model.best_model_error:.3f} and "
            f"test error {forecast_error:.3f}"
        )


def test_date_ranges(
    train_data: pd.DataFrame,
    dataset_name: str,
    seasonality: Union[int, List[int]],
    start_date: dt.datetime,
    end_date: dt.datetime,
    models: Union[List[str], Tuple[str]] = None,
):
    print(f"Starting date range test for dataset {dataset_name} with models {models} test "
          f"start date {start_date} test end date {end_date}")
    if models is not None:
        model = AutoTS(seasonal_period=seasonality, verbose=2, model_names=models)
    else:
        model = AutoTS(seasonal_period=seasonality, verbose=2)
    model.fit(data=train_data, series_column_name="ts")
    forecast = pd.Series(data=model.predict(start_date, end_date), name="forecast")
    print("\tDate range test success")


if __name__ == "__main__":
    start_time = dt.datetime.now()
    airline = pd.read_csv("../examples/airline_passengers/AirPassengers.csv")
    airline["Month"] = pd.to_datetime(airline["Month"])
    airline = airline.set_index("Month")
    airline = airline.rename({"Passengers": "ts"}, axis="columns")
    airline_train, airline_test = airline[:-12], airline[-12:]

    shampoo = pd.read_csv("../examples/shampoo/shampoo.csv")
    shampoo["Month"] = pd.to_datetime(shampoo["Month"])
    shampoo = shampoo.set_index("Month")
    shampoo = shampoo.rename({"Sales": "ts"}, axis="columns")
    shampoo_train, shampoo_test = shampoo[:30], shampoo[30:]

    gasoline = pd.read_csv("../examples/gasoline/gasoline.csv")
    gasoline["week"] = pd.to_datetime(gasoline["week"])
    gasoline = gasoline.set_index("week").sort_index()
    gasoline = gasoline.rename({"barrels_of_gasoline": "ts"}, axis="columns")
    gasoline_train, gasoline_test = gasoline[-156:-52], gasoline[-52:]  # limit 2 years to speed up modeling

    model_sets = [["auto_arima"], ["exponential_smoothing"], ["tbats"], None]
    airline_test_dates = [
        [airline_train.index[-2], airline_test.index[-1]],
        [airline_train.index[-2] + relativedelta(days=-1), airline_test.index[-1]],
        [airline_test.index[0], airline_test.index[-1]],
    ]
    gasoline_test_dates = [
        [gasoline_train.index[-20], gasoline_test.index[-8]],
        [gasoline_train.index[-2], gasoline_test.index[-1]],
        [gasoline_train.index[-2] + relativedelta(days=-1), gasoline_test.index[-1]],
        [gasoline_test.index[0], gasoline_test.index[-1]],
    ]

    seasonality_tests(airline)
    test_accuracy(airline_train, airline_test, "shampoo", 6)
    test_accuracy(gasoline_train, gasoline_test, "gasoline", 52)

    for model_set in model_sets:
        for dates in airline_test_dates:
            test_date_ranges(airline_train, "airline", 12, start_date=dates[0], end_date=dates[1], models=model_set)

    for model_set in model_sets:
        for dates in gasoline_test_dates:
            test_date_ranges(gasoline_train, "gasoline", 52, start_date=dates[0], end_date=dates[1], models=model_set)
    end_time = dt.time()
    print(f"Completed testing in {dt.datetime.now() - start_time:.1} seconds")
