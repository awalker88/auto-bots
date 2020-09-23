import pickle as pkl
import pandas as pd

from AutoTS import AutoTS

from time import time


def main():
    split_dfs = clean_data()
    config = {'prediction_splitters': ['geography_code', 'market_name', 'dual_unit_name', 'dual_sub_unit_name']}
    start_time = time()

    for i, tdf in enumerate(split_dfs):
        atomic_id = ', '.join([f'{splitter}: {str(tdf[splitter].iloc[0])}'
                               for splitter in config['prediction_splitters']])
        if len(tdf) < 10:
            print(f"{i}/{len(split_dfs)} | {atomic_id} | too short {len(tdf)}")
        elif tdf['expense_plan_amount'].mean() < 1000 and tdf['expense_plan_amount'].max() < 10_000:
            print(f"{i}/{len(split_dfs)} | {atomic_id} | too poor {tdf['expense_plan_amount'].mean():.0f}")
        elif tdf.index[-1] < pd.to_datetime('2020-7-1'):
            print(f"{i}/{len(split_dfs)} | {atomic_id} | ends too early len={len(tdf)} last_date={tdf.index[-1]}")
        else:
            model = AutoTS()
            model.fit(tdf, series_column_name='expense_plan_amount', exogenous=['is_covid'])

    end_time = time()
    print(f'time to model: {end_time - start_time:.1f} seconds')


def clean_data():
    split_dfs = pkl.load(open('smol_split_dfs.pkl', 'rb'))
    split_dfs = [test_df.set_index('finance_start_date', drop=False) for test_df in split_dfs]
    split_dfs = add_zero_rows(split_dfs, start_date='2018-1-1', end_date='2020-7-1')

    split_dfs = [is_covid(df) for df in split_dfs]

    return split_dfs


def add_zero_rows(split_dfs: list, start_date: str, end_date: str):
    date_range = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='MS')
    zero_df = pd.DataFrame({'zeros': [0] * len(date_range)}, index=date_range)

    def add_zero_rows_helper(df: pd.DataFrame):
        df = df.merge(zero_df, how='outer', left_index=True, right_index=True).drop('zeros', axis='columns')
        df['expense_plan_amount'] = df['expense_plan_amount'].fillna(0)
        df['finance_start_date'] = df['finance_start_date'].fillna(df.index.to_series())
        df = df.fillna(method='backfill')
        df = df.fillna(method='ffill')

        return df

    split_dfs = [add_zero_rows_helper(df) for df in split_dfs]

    return split_dfs


def is_covid(df: pd.DataFrame):
    df['is_covid'] = 0
    df.loc[df['finance_start_date'] > '2020-2-1', 'is_covid'] = 1
    return df


if __name__ == '__main__':
    main()
