# auto-ts

AutoTS is an easy-to-use time series forecasting model that does all the model selection work for you

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
<sub><sup><sub><sup>It's a lot of AutoTS's, I know</sup></sub></sup></sub>

You may need to add some more to the import statement if you put the AutoTS folder inside another folder.
For example, if you put it in a folder named "src" inside your repo, it might need to look more like this:
```python
from src.AutoTS.AutoTS import AutoTS
```

# Quickstart

For a good intro to how to use the model, check out [this example]()