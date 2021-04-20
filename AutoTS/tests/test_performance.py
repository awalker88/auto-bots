import pandas as pd
import matplotlib.pyplot as plt
from time import time

from AutoTS.AutoTS import AutoTS

print('starting test_performance')

airline_data = pd.read_csv('/Users/andrewwalker/PycharmProjects/auto-ts/examples/airline_passengers/AirPassengers.csv')
airline_data['Month'] = pd.to_datetime(airline_data['Month'])
airline_data = airline_data.set_index('Month')
airline_data_train, airline_data_test = airline_data[:120], airline_data[120:]

m_start = time()
model = AutoTS(seasonal_period=12,
               verbose=True
               # model_names=['auto_arima']
               )
model.fit(
    data=airline_data_train,
    series_column_name='Passengers'
          )

print(f'AutoTS found the best model for your data to be {model.fit_model_type}, with a '
      f'{model.error_metric} of {model.best_model_error:.2f}')

preds = model.predict(start_date=airline_data_test.index[0], end_date=airline_data.index[-1])
m_end = time()
print(f'Time to do modeling was {m_end - m_start:.2f}s')

# plt.plot(airline_data_train, c='blue')
# plt.plot(preds, c='green')
# plt.show()
