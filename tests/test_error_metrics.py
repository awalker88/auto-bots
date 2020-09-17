import pandas as pd

from error_metrics import mase


def test_mase():
    mase_test_df = pd.DataFrame(data={'prediction': [12, 18, 14, 16], 'actuals': [10, 20, 15, 15]})
    avg_shifted_error = (abs(20 - 10) + abs(15 - 20) + abs(15 - 15)) / 3
    mase_test_df['scaled_prediction_error'] = \
        abs(mase_test_df['prediction'] - mase_test_df['actuals']) / avg_shifted_error
    correct_error = mase_test_df['scaled_prediction_error'].mean()

    test_error = mase(mase_test_df, 'prediction', 'actuals')

    print(mase_test_df)
    print(avg_shifted_error, correct_error, test_error)

    assert correct_error == test_error


if __name__ == '__main__':
    test_mase()