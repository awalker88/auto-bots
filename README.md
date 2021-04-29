# auto-ts

AutoTS is an easy-to-use time series forecasting model that does all the model selection work for you.

# Installation

First, make sure you have installed the following packages (make sure to get the right versions when specified)
```
pandas
pmdarima==1.71
statsmodels==0.11.1
tbats=1.1.0
```
Put the AutoTS folder in your repository. Then, to use the AutoTS model in your code, import it like so:
```python
from AutoTS.AutoTS import AutoTS
```
<sub><sup>It's a lot of AutoTS's, I know</sup></sub>

You may need to add some more to the import statement if you put the AutoTS folder inside another folder.
For example, if you put it in a folder named "src" inside your repo, it might need to look more like this:
```python
from src.AutoTS.AutoTS import AutoTS
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
