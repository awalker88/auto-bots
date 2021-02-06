import pandas as pd


def check_models(models):
    if type(models) not in [tuple, list]:
        raise TypeError('`models` argument must a list or tuple')

    valid_models = ['auto_arima', 'exponential_smoothing', 'tbats', 'ensemble']
    if len(models) == 0:
        raise ValueError(f'`models` argument must contain at least one of {valid_models}')

    invalid_models = [model for model in models if model not in valid_models]
    if len(invalid_models) > 0:
        raise ValueError(f'The following models are not supported: {invalid_models}')

    if len(models) <= 2 and 'ensemble' in models:
        raise ValueError('If you wish to have `ensemble` be a candidate model, you must specify at '
                         'least two additional valid models')


def check_datetime_index(series_df: pd.DataFrame):
    if not isinstance(series_df.index, pd.DatetimeIndex):
        raise TypeError('The index of your dataframe must be a series of datetimes')