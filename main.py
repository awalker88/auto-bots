from dateutil.relativedelta import relativedelta
import pickle as pkl
from time import time

import pandas as pd
import numpy
from scipy.stats import zscore

from AutoTS import AutoTS
from utils.log_helper import get_logger, log_setup

log = get_logger(__name__)
# numpy.seterr('raise')


def forecast(environment: str):
    start_time = time()
    config = make_config()
    # raw_data = pd.read_csv('../../data/forecast input/epm_travel_pull.csv', low_memory=False)
    log.info('read in csv')
    test_name = '6m_seasonality_3Q19'

    # time_sks = pd.read_csv(f'../../data/SK Data/{environment}_period.csv').\
    #     rename({'DATE': 'finance_month_start_date'}, axis='columns')

    log.info('preparing data')
    log.info(f'splitting on {config["forecast_splitters"]}')
    # split_dfs, full_data = prepare_data(raw_data, config)
    split_dfs, full_data = pkl.load(open('split_dfs.pkl', 'rb')), pkl.load(open('full_data.pkl', 'rb'))
    # pkl.dump(split_dfs, open('split_dfs.pkl', 'wb'))
    # pkl.dump(full_data, open('full_data.pkl', 'wb'))
    # split_dfs = [add_forecast_dates(df, time_sks, config) for df in split_dfs]

    log.info('starting forecasts')
    forecast_df = pd.concat([add_forecast(df, f'{i + 1}/{len(split_dfs)}', config) for i, df in enumerate(split_dfs[:])])
    forecast_output = forecast_df[~forecast_df['forecast'].isna()].copy()
    forecast_output['test_name'] = test_name

    forecast_csv = pd.read_csv('forecast_output_multi.csv')
    forecast_csv = forecast_csv[forecast_csv['test_name'] != test_name]
    forecast_csv = pd.concat([forecast_csv, forecast_output])

    forecast_csv.to_csv('forecast_output_multi.csv', index=False, date_format='%M/%d/%Y')

    end_time = time()
    log.info(f'finished forecasting process in {end_time - start_time:.1f} seconds')


def make_config():
    forecast_start_date = pd.to_datetime('2020-10-1')
    forecast_end_date = pd.to_datetime('2020-12-1')
    config = {
        'forecast_splitters':
            ['geography_code', 'market_code', 'dual_unit_name', 'dual_sub_unit_name'],
        'forecast_start_date': forecast_start_date,
        'forecast_end_date': forecast_end_date,
        'forecast_dates':
            pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS').tolist()
    }

    return config


def prepare_data(raw_data: pd.DataFrame, config: dict) -> (list, pd.DataFrame):

    ## data cleaning
    raw_data.columns = map(str.lower, raw_data.columns)
    raw_data['finance_month_start_date'] = pd.to_datetime(raw_data['finance_month_start_date'])
    raw_data = raw_data.sort_values(['finance_month_start_date', 'geography_code', 'market_code', 'country_name',
                                     'dual_unit_name', 'dual_sub_unit_name', 'dual_division_name']).reset_index(drop=True)

    split_dfs = split_data(raw_data, config)

    split_dfs = [test_df.set_index('finance_month_start_date', drop=False) for test_df in split_dfs]
    split_dfs = [add_zero_rows(df) for df in split_dfs]
    split_dfs = [add_features(df) for df in split_dfs]
    split_dfs = [dampen_anomalies(df, 'expense_plan_amount', config) for df in split_dfs]

    # todo: if any forecast dates are past current month, add forecast dates

    return split_dfs, raw_data


def add_forecast_dates(df, time_sks: pd.DataFrame, config: dict) -> pd.DataFrame:
    log.info('adding forecast dates')
    df['forecast_month'] = df.apply(lambda x: 1 if pd.to_datetime(x['finance_month_start_date']) in
                                                   config['forecast_dates'] else 0, axis='columns')

    next_month = df['finance_month_start_date'].iloc[-1]
    while next_month < config['forecast_end_date']:
        next_month = next_month + relativedelta(months=+1)
        next_row = {'finance_month_start_date': next_month,
                    'year_num': next_month.year,
                    'quarter_num': next_month.quarter,
                    'month_num': next_month.month,
                    'forecast_month': 1}
        df = pd.concat([df, pd.DataFrame(next_row, index=[0])])

    df = df.drop('sk_finance_time_period_month', axis='columns')
    df = df.merge(time_sks[['finance_month_start_date', 'sk_finance_time_period_month']], how='left',
                  on=['finance_month_start_date'])

    # ok that we don't check for things in `next_row` since they should have nothing empty to fill in
    df = df.ffill()

    return df


def split_data(df: pd.DataFrame, config: dict) -> list:
    """
    chunks up df by the columns specified in splitters

    Ex. If splitters is ['sk_sub_unit', 'geography_code'] and there are 2 unique sub units and
    3 unique geography codes, then this function will return a list with 6 dataframes:
    [subunit 1/geo 1, subunit 1/geo 2, subunit 1/geo 3, subunit 2/geo 1, subunit 2/geo 2, subunit 2/geo 3]

    NOTE: If you add any columns to the data pull that are hierarchical (ex. geography is split up
    into markets), you will need to use remove_absent_splitters on the lower levels before larger ones if
    they are not specified in `config['forecast_splitters']`. This is already done for
    market-geography.

    :param df: expense data with minors/elements as columns
    :param config: project level configuration dictionary
    :return: a list of split dataframes
    """
    splitters = config['forecast_splitters'].copy()

    def remove_absent_splitters(pre_split_df: pd.DataFrame, splitters_to_remove: list) -> pd.DataFrame:
        pre_split_df = pre_split_df.drop(splitters_to_remove, axis='columns', errors='ignore')
        pre_split_df = pre_split_df.groupby(
            [col for col in pre_split_df.columns if col not in ['expense_plan_amount']]).sum().reset_index()
        pre_split_df = pre_split_df.sort_values(['finance_month_start_date'])

        return pre_split_df

    if 'sk_country' not in splitters and 'country_name' not in splitters and 'country_code' not in splitters:
        df = remove_absent_splitters(df, ['sk_country', 'country_name', 'country_code'])

    if 'sk_geo_market' not in splitters and 'market_code' not in splitters:
        df = remove_absent_splitters(df, ['sk_geo_market', 'market_code'])

    if 'geography_name' not in splitters and 'geography_code' not in splitters:
        df = remove_absent_splitters(df, ['geography_name', 'geography_code'])

    if 'sk_division' not in splitters and 'dual_division_name' not in splitters:
        df = remove_absent_splitters(df, ['sk_division', 'dual_division_name'])

    if 'sk_sub_unit' not in splitters and 'dual_sub_unit_name' not in splitters:
        df = remove_absent_splitters(df, ['sk_sub_unit', 'dual_sub_unit_name'])

    if 'dual_unit_name' not in splitters:
        df = remove_absent_splitters(df, ['dual_unit_name'])

    # iteratively split dataframe into smaller and smaller chunks
    to_be_splits = [df]
    while splitters:
        split_col = splitters.pop(0)
        after_split = []
        for to_be_split in to_be_splits:
            # .get_group(unique_value) allows us to select rows with a specific value (ex. get all
            # AP rows, then it will get all EM rows, etc.)
            splitted = [to_be_split.groupby(split_col).get_group(unique_value)
                        for unique_value in to_be_split[split_col].unique().tolist()]
            after_split.extend(splitted)

        # after we split on this split_col, make before_splits ready to be split on by the next one
        to_be_splits = after_split

    return to_be_splits


def add_zero_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    example: TW-WWT-CorpHQ Consol-IBM Other
    :param df:
    :return:
    """
    date_range = pd.date_range(df.index[0], df.index[-1], freq='MS')
    zero_df = pd.DataFrame({'zeros': [0] * len(date_range)}, index=date_range)
    df = df.merge(zero_df, how='outer', left_index=True, right_index=True, indicator=True).drop('zeros', axis='columns')
    df['expense_plan_amount'] = df['expense_plan_amount'].fillna(0)
    df['finance_month_start_date'] = df['finance_month_start_date'].fillna(df.index.to_series())
    df = df.fillna(method='backfill')
    df = df.fillna(method='ffill')

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_covid'] = df['finance_month_start_date'].apply(lambda x: 1 if x >= pd.to_datetime('2020-4-1')
    else 0)

    return df


def dampen_anomalies(df: pd.DataFrame, amount_col: str, config: dict) -> pd.DataFrame:
    # if all rows are 0, can't define what an anomaly is. same if we only have one row
    if (len(df[df[amount_col] == 0]) == len(df)) or (len(df) == 1):
        return df

    df_anom = df.copy()
    df_anom['average'] = df_anom[amount_col].mean()
    df_anom['std'] = df_anom[amount_col].std()
    df_anom['z_score'] = zscore(df_anom[amount_col].to_list(), nan_policy='omit')

    def dampen_anomalies_helper(row: pd.DataFrame, z_score_dampener: float = 2) -> float:
        dampened_value = row[amount_col]

        if row['z_score'] > 3 and abs(row[amount_col] - row['average']) > 1000:
            dampened_value = row['average'] + z_score_dampener * row['std']
        elif row['z_score'] < -3 and abs(row[amount_col] - row['average']) > 1000:
            dampened_value = row['average'] - z_score_dampener * row['std']

        if dampened_value != row[amount_col]:
            specs = ' | '.join([row['finance_month_start_date'].strftime('%Y-%m')] +
                               [f"{splitter} = {row[splitter]}" for splitter in config['forecast_splitters']])
            print(f"Value [${int(row[amount_col]):,.0f}] was replaced with [${dampened_value:,.0f}] for having a z-score "
                  f"of [{row['z_score']:.1f}] vs. average [${int(row['average']):,.0f}] / "
                  f"standard deviation [${int(row['std']):,.0f}] in {specs}")

        return dampened_value

    # save version of expense column before overwriting with dampened values
    df_anom[f'{amount_col}_pre_dampen'] = df_anom[amount_col]
    df_anom[amount_col] = df_anom.apply(dampen_anomalies_helper, axis='columns')

    df_anom = df_anom.drop(columns=['z_score', 'average', 'std'])

    return df_anom


def add_forecast(input_df: pd.DataFrame, forecast_count: str, config: dict):
    atomic_id = ', '.join([f'{splitter}: {str(input_df[splitter].iloc[0])}'
                           for splitter in config['forecast_splitters']])
    forecast_date_range = pd.date_range(start=config['forecast_start_date'], end=config['forecast_end_date'], freq='MS')
    training_data = input_df[input_df['finance_month_start_date'] < config['forecast_start_date']].copy()
    log.info(f'{forecast_count} making model for {atomic_id} | Training series length = {len(training_data)}')

    # 10/1/20 don't know why this works but it fixes an "underflow errror in true_divide" deep in the statsmodels package, try removing it in the future
    training_data['expense_plan_amount'] = round(training_data['expense_plan_amount'], 0)

    # if less than 9 months of data, predict average todo: look more heavily into this, right now it's just to fix some bugs
    if len(training_data) < 9:
        forecasts = pd.Series(
            data=[training_data.iloc[-12:, :]['expense_plan_amount'].mean()] * len(forecast_date_range),
            index=forecast_date_range, name='forecast')
        model_type = 'average'
        log.info('\tUsing average method due to insufficient data')

    # if the last 6 months have 0 expense, predict 0
    elif len(input_df.iloc[-4:, :].query('expense_plan_amount == 0')) == 4:
        forecasts = pd.Series([0] * len(forecast_date_range), index=forecast_date_range, name='forecast')
        model_type = 'zeros'
        log.info('\tUsing zero method')

    # if percentage of months with 0 expense is greater than 67%, just predict average
    elif (len(training_data[training_data['expense_plan_amount'] == 0]) / len(training_data)) > .67:
        forecasts = pd.Series(
            data=[training_data.iloc[-12:, :]['expense_plan_amount'].mean()] * len(forecast_date_range),
            index=forecast_date_range, name='forecast')
        model_type = 'average'
        log.info('\tUsing average method due to >67% records == 0')

    # if it's small dollars, just predict average
    elif training_data['expense_plan_amount'].mean() < 1000 and training_data['expense_plan_amount'].std() < 5_000:
        forecasts = pd.Series(
            data=[training_data.iloc[-12:, :]['expense_plan_amount'].mean()] * len(forecast_date_range),
            index=forecast_date_range, name='forecast')
        model_type = 'average'
        log.info('\tUsing average method due to small dollars')

    # if we have a super small standard deviation, just predict average (this helps avoid errors in
    # AutoTS dependencies, which sometimes assumes series to be non-constant
    elif training_data['expense_plan_amount'].std() < 10:
        forecasts = pd.Series(
            data=[training_data.iloc[-12:, :]['expense_plan_amount'].mean()] * len(forecast_date_range),
            index=forecast_date_range, name='forecast')
        model_type = 'average'
        log.info('\tUsing average method due to small standard deviation')

    # ends too early
    elif training_data.index[-1] + relativedelta(months=+1) < pd.to_datetime(config['forecast_start_date']):
        log.info('\tEnds too early')
        return None
        # raise AttributeError(f"{atomic_id} | ends too early len={len(training_data)} last_date={training_data.index[-1]}")

    else:
        model = AutoTS(model_names=['auto_arima', 'exponential_smoothing', 'ensemble'], verbose=True, seasonal_period=2)
        model.fit(training_data, series_column_name='expense_plan_amount',
                  # exogenous=['is_covid']
                  )

        # predict_exogenous = input_df[(input_df.index >= config['forecast_start_date']) & (input_df.index <= config['forecast_end_date'])]['is_covid']
        predict_exogenous = pd.DataFrame({'is_covid': [1] * len(config['forecast_dates'])})

        forecasts = pd.Series(
            data=model.predict(config['forecast_start_date'], config['forecast_end_date'],
                               # predict_exogenous
                               ),
            index=forecast_date_range, name='forecast')
        model_type = model.fit_model_type

    output_df = input_df.merge(forecasts, how='outer', left_index=True, right_index=True)
    output_df['model'] = model_type

    return output_df


if __name__ == '__main__':
    log_setup()
    forecast('preprod')
