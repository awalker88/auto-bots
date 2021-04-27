from typing import Union, List, Tuple

import pandas as pd

from AutoTS.AutoTS import AutoTS
from AutoTS.utils.error_metrics import mase


def seasonality_tests(data: pd.DataFrame):
    possible_seasonalities = [3, [3], [12, 3]]
    for seasonality in possible_seasonalities:
        model = AutoTS(seasonal_period=seasonality)
        model.fit(data=data, series_column_name="ts")
        print(f"Test passed with seasonality = {seasonality}")


def test_accuracy(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    dataset_name: str,
    seasonality: Union[int, List[int]],
    models: Union[List[str], Tuple[str]] = None,
):
    print(f"Starting accuracy test for dataset {dataset_name}")
    if models is not None:
        model = AutoTS(seasonal_period=seasonality, verbose=2, model_names=models)
    else:
        model = AutoTS(seasonal_period=seasonality, verbose=2)
    model.fit(data=train_data, series_column_name="ts")

    forecast = pd.Series(
        data=model.predict(start_date=test_data.index[0], end_date=test_data.index[-1]),
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
        assert forecast_error < 1
        print(
            f"Passed airline accuracy test with train error = {model.best_model_error:.3f} and "
            f"test error {forecast_error:.3f}"
        )

    elif dataset_name == "gasoline":
        pass


if __name__ == "__main__":
    # airline = pd.read_csv('../examples/airline_passengers/AirPassengers.csv')
    # airline['Month'] = pd.to_datetime(airline['Month'])
    # airline = airline.set_index('Month')
    # airline = airline.rename({'Passengers': 'ts'}, axis='columns')
    # airline_train, airline_test = airline[:120], airline[120:]
    #
    # shampoo = pd.read_csv('../examples/shampoo/shampoo.csv')
    # shampoo['Month'] = pd.to_datetime(shampoo['Month'])
    # shampoo = shampoo.set_index('Month')
    # shampoo = shampoo.rename({'Sales': 'ts'}, axis='columns')
    # shampoo_train, shampoo_test = shampoo[:30], shampoo[30:]
    #
    # sunspots = pd.read_csv('../examples/sunspots/sunspots.csv')
    # sunspots['Month'] = pd.to_datetime(sunspots['Month'])
    # sunspots = sunspots.set_index('Month')
    # sunspots = sunspots.rename({'Sunspots': 'ts'}, axis='columns')
    # sunspots_train, sunspots_test = sunspots[:2256], sunspots[2256:]

    gasoline = pd.read_csv("../examples/gasoline/gasoline.csv")
    gasoline["week"] = pd.to_datetime(gasoline["week"])
    gasoline = gasoline.set_index("week").sort_index()
    gasoline = gasoline.rename({"barrels_of_gasoline": "ts"}, axis="columns")
    gasoline_train, gasoline_test = gasoline[:-104], gasoline[-104:]

    # seasonality_tests(airline)
    # test_accuracy(airline_train, airline_test, 'airline', 12)
    # test_accuracy(shampoo_train, shampoo_test, 'shampoo', 6)
    # test_accuracy(sunspots_train, sunspots_test 'sunspots', 6)
    test_accuracy(gasoline_train, gasoline_test, "gasoline", 52, models=["tbats"])
