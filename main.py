import pickle as pkl
import pandas as pd
from scipy.stats import zscore

from AutoTS import AutoTS

from time import time

# todo: come up with some way to smooth out patterns where there's a huge spike one month that's offset the next


def main():
    config = {'prediction_splitters': ['geography_code', 'market_name', 'dual_unit_name', 'dual_sub_unit_name']}
    start_time = time()
    # split_dfs = clean_data(config)
    # pkl.dump(split_dfs, open('cleaned_split_dfs.pkl', 'wb'))
    split_dfs = pkl.load(open('cleaned_split_dfs.pkl', 'rb'))
    results = []
    for i, tdf in enumerate(split_dfs):
        atomic_id = ', '.join([f'{splitter}: {str(tdf[splitter].iloc[0])}'
                               for splitter in config['prediction_splitters']])
        if len(tdf[tdf['expense_plan_amount'] != 0]) < 12:
            print(f"{i}/{len(split_dfs)} | {atomic_id} | too few non-zero data {len(tdf[tdf['expense_plan_amount'] != 0])}/{len(tdf)}")
        elif len(tdf.iloc[-6:, :].query('expense_plan_amount == 0')) == 6:
            print(f"{i}/{len(split_dfs)} | {atomic_id} | at least the last 6 rows are 0")
        elif tdf['expense_plan_amount'].mean() < 1000 and tdf['expense_plan_amount'].std() < 10_000:
            print(f"{i}/{len(split_dfs)} | {atomic_id} | too poor {tdf['expense_plan_amount'].mean():.0f}")
        elif tdf.index[-1] < pd.to_datetime('2020-2-1'):
            print(f"{i}/{len(split_dfs)} | {atomic_id} | ends too early len={len(tdf)} last_date={tdf.index[-1]}")
        else:
            print(f"{i}/{len(split_dfs)} len={len(tdf)} lenN0={len(tdf[tdf['expense_plan_amount'] != 0])} avg={tdf['expense_plan_amount'].mean():.1f}")
            model = AutoTS(seasonal_period=6, model_names=('auto_arima', 'exponential_smoothing', 'tbats', 'ensemble'), verbose=True)
            model.fit(tdf, series_column_name='expense_plan_amount')
            p = model.predict(start_date=pd.to_datetime('2020-1-1'), end_date=pd.to_datetime('2020-2-1'))
            results.append([atomic_id, model.fit_model_type, model.best_model_error])

    results_df = pd.DataFrame(results, columns=['model_id', 'model_type', 'error'])
    results_df.to_csv('results_df.csv', index=False)
    print('avg error =', results_df[results_df['error'] < 10]['error'].mean())
    end_time = time()
    print(f'time to model: {end_time - start_time:.1f} seconds')


def clean_data(config) -> list:
    split_dfs = pkl.load(open('smol_split_dfs.pkl', 'rb'))
    split_dfs = [test_df.set_index('finance_start_date', drop=False) for test_df in split_dfs]
    split_dfs = [add_zero_rows(df) for df in split_dfs]

    split_dfs = [is_covid(df) for df in split_dfs]

    last_normal_month = pd.to_datetime('2020-3-1')
    split_dfs = [df[df['finance_start_date'] < last_normal_month] for df in split_dfs]
    before_dfs = len(split_dfs)
    split_dfs = [df for df in split_dfs if df.empty is False]
    after_dfs = len(split_dfs)
    print(f"Filtering for dates before {last_normal_month} removed {before_dfs - after_dfs} dfs")

    split_dfs = [remove_anomalies(df, 'expense_plan_amount', config) for df in split_dfs]

    return split_dfs


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
    df['finance_start_date'] = df['finance_start_date'].fillna(df.index.to_series())
    df = df.fillna(method='backfill')
    df = df.fillna(method='ffill')

    return df


def is_covid(df: pd.DataFrame):
    df['is_covid'] = 0
    df.loc[df['finance_start_date'] > '2020-2-1', 'is_covid'] = 1

    return df


def remove_anomalies(df: pd.DataFrame, amount_col: str, config: dict) -> pd.DataFrame:
    # if all rows are 0, can't define what an anomaly is. same if we only have one row
    if (len(df[df[amount_col] == 0]) == len(df)) or (len(df) == 1):
        return df

    df_anom = df.copy()
    df_anom['average'] = df_anom[amount_col].mean()
    df_anom['std'] = df_anom[amount_col].std()
    df_anom['z_score'] = zscore(df_anom[amount_col].to_list(), nan_policy='omit')

    def dampen_anomaly(row: pd.DataFrame, z_score_dampener: float = 2) -> float:
        dampened_value = row[amount_col]

        if row['z_score'] > 3 and abs(row[amount_col] - row['average']) > 1000:
            dampened_value = row['average'] + z_score_dampener * row['std']
        elif row['z_score'] < -3 and abs(row[amount_col] - row['average']) > 1000:
            dampened_value = row['average'] - z_score_dampener * row['std']

        if dampened_value != row[amount_col]:
            specs = ' | '.join([row['finance_start_date'].strftime('%Y-%m')] +
                               [f"{splitter} = {row[splitter]}" for splitter in config['prediction_splitters']])
            print(f"Value [${int(row[amount_col]):,.0f}] was replaced with [${dampened_value:,.0f}] for having a z-score "
                  f"of [{row['z_score']:.1f}] vs. average [${int(row['average']):,.0f}] / "
                  f"standard deviation [${int(row['std']):,.0f}] in {specs}")

        return dampened_value

    df_anom[amount_col] = df_anom.apply(dampen_anomaly, axis='columns')

    df_anom = df_anom.drop(columns=['z_score', 'average', 'std'])

    return df_anom


def flag_for_average(df):
    pass


if __name__ == '__main__':
    main()
