# Contribution guide

This repository is designed to make it as easy as possible to add new model or library.
First, look at the structure of the project. It includes several levels of abstraction. If you want to add a model, you need to look 
at the folder with [models](../pytsbe/models) (obligatory) and [serializers](../pytsbe/store) (optional).

## Implement Forecaster
`Forecaster` is a class that generates time series forecasts.

Example of implementation: [ARIMAForecaster](../pytsbe/models/pmdarima_forecaster.py)

You will need to implement all the methods listed below.

### Method `__init__`

`**params` - parameters for model or library you want to set.

### Method `fit`

`historical_values` - historical data for model fitting. Example of expected table:

| datetime   | value |
| :--------: | :---: |
| 01-01-2022 | 254   |
| 02-01-2022 | 223   |

`forecast_horizon` - number of elements to predict.

`**kwargs` - additional parameters.

### Method `predict`
The method should allow to make a prediction for the future on the basis of
historical values. The forecast horizon will not differ from that during fit stage.

`historical_values` - pandas Dataframe with historical values that is the last 
known historical values at the time the forecast was generated.

`forecast_horizon` - number of elements to predict.

`**kwargs` - additional parameters.

The method returns a special dataclass `ForecastResults`. Check the fields in this class, 
in addition to predictions as a numpy array, it is able to pass any additional information 
you want to keep during the experiments (it can be handled in the serializer).

You should then think of a name for your model and 
add to the dictionary in the [`Validator` class](../pytsbe/validation/validation.py).

## Implement Serializer (optional)
Serializer is a class for saving the results of experiments.

Example of implementation: [FedotSerializer](../pytsbe/store/fedot_serializer.py)

It is not necessary to implement this class. For new models, the default 
serializer will automatically run and save the model predictions. But if you
want to store additional information (for example), such as a model description, 
you can add own implementation.
