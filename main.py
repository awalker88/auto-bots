import pandas as pd

from AutoTS.AutoTS import AutoTS
from AutoTS.utils.log_helper import get_logger, log_setup

log = get_logger(__name__)
# numpy.seterr('raise')

if __name__ == '__main__':
    log_setup()
    # forecast('preprod')

    airline_data = pd.read_csv('examples/airline_passengers/AirPassengers.csv')
    airline_data['Month'] = pd.to_datetime(airline_data['Month'])
    airline_data = airline_data.set_index('Month')
    airline_data.head()

    model = AutoTS(seasonal_period=12)
    model.fit(airline_data, 'Passengers')
    preds = model.predict(start_date='1961-1-1', end_date='1961-2-1')
    print(preds)