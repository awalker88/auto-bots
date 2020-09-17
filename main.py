import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

from AutoTS import AutoTS

from time import time


def main():
    split_dfs = pkl.load(open('split_dfs.pkl', 'rb'))

    test_df: pd.DataFrame = split_dfs[0]
    test_df = test_df.set_index('finance_start_date', drop=False)

    start_time = time()

    model = AutoTS()

    model = model.fit(test_df, x='expense_plan_amount', exogenous=['is_covid'])

    end_time = time()
    print(f'time to model: {end_time - start_time:.1f} seconds')


if __name__ == '__main__':
    main()
