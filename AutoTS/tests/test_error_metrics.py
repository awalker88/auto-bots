import pandas as pd

from AutoTS.utils.error_metrics import mase, mse


def test_mase(tolerance: float = 0.01):
    """ Tests Mean Absolute Squared Error"""
    mase_test_df = pd.DataFrame(data={'prediction': [12, 18, 14, 16], 'actuals': [10, 20, 15, 15]})
    avg_shifted_error = (abs(20 - 10) + abs(15 - 20) + abs(15 - 15)) / 3
    mase_test_df['scaled_prediction_error'] = \
        abs(mase_test_df['prediction'] - mase_test_df['actuals']) / avg_shifted_error
    correct_error = mase_test_df['scaled_prediction_error'].mean()

    test_error = mase(mase_test_df, 'prediction', 'actuals', 1)

    try:
        assert abs(correct_error - test_error) < tolerance
        print('Passed MASE test')
    except AssertionError:
        raise AssertionError(f'MASE test failed: Difference between {correct_error} and {test_error} '
                             f'is greater than {tolerance}')


def test_mse(tolerance: float = 0.01):
    mse_test_df = pd.DataFrame(data={'prediction': [12, 18, 14, 16], 'actuals': [10, 20, 15, 15]})
    correct_error = ((12 - 10) ** 2 + (18 - 20) ** 2 + (14 - 15) ** 2 + (16 - 15) ** 2) / 4
    test_error = mse(mse_test_df, 'prediction', 'actuals')

    try:
        assert abs(correct_error - test_error) < tolerance
        print('Passed MSE test')
    except AssertionError:
        raise AssertionError(f'MSE test failed: Difference between {correct_error} and {test_error} '
                             f'is greater than {tolerance}')


if __name__ == '__main__':
    test_mase()
    test_mse()
