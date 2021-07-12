# auto-bots

`auto-bots` is an easy-to-use time series forecasting package that does all the model selection work for you.

# Installation
You can install `auto-bots` with a simple
```commandline
pip install auto-bots
```

Then, to use the AutoTS model in your code, import it like so:
```python
from auto_bots.AutoTS import AutoTS
```


# Quickstart

AutoTS follows sci-kit learn's `model.fit()`/`model.predict()` paradigm. The only requirement of your
data is that it must be a pandas dataframe with a datetime index. Given such a dataframe, here is how
to train your model and make predictions:

```python
model = AutoTS()

model.fit(data, series_column_name='passengers')
model.predict(start=pd.to_datetime('1960-1-1'), end=pd.to_datetime('1960-12-1'))
```

### Tips/Tricks/Things to know
- Since you provide the name of the time series column during fit, the dataframe provided 
during fit can contain as many extra columns as you like and the model will ignore them. No need to do
a bunch of filtering before training!
- You can have the model predict in-sample by setting the `start_date` equal to a date inside the data given during fit.


For a more thorough introduction, check out [this example](examples/airline_passengers/airline_example.ipynb)
