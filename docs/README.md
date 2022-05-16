# Common documentation description 

Welcome to the main documentation page of **pytsbe** library. 

---

## Useful features (why you should try pytsbe)

The module allows a variety of experiments to be carried 
out on different time series, as well as flexible configuration
of experiment conditions and library parameters.

The evaluation process is divided into two parts: running
experiments and calculating metrics. During the first stage,
csv files with model predictions are generated, 
as well as json files with `fit` and `predict` method execution times for 
each model / library. Once the experiments have been completed, the library 
functionality can be used to generate reports, calculate metrics and plot
graphs. Since predictions are saved, as well as additional information if desired, 
it is always possible to calculate further metrics and construct new graphs (especially useful
when writing scientific papers).


**The advantages of this module are:**
- Various time series and libraries have already been integrated into 
  the repository, and the wrappers have been tested and are ready for use

<img src="./images/features_1.png" width="700"/> 

- Ability to perform validation both on the last segment 
  of the time series and to use in-sample forecasting
- While it is running, the algorithm saves the conditions of the experiment
  so that it can be reproduced in the future (saves a configuration file) 
- The algorithm will continue to work even if the model fails during the calculations. 
  Then the result for this case will not be generated and the algorithm moves on to the next 
- Ability to restart experiments if they were previously stopped 
  for unexpected reasons. In this case, the algorithm will check which cases have already been calculated and start from where it left off
- If you re-run the experiment in the existing working directory and change the experiment coditions, the module will detect the problem
  (compare it to the existing configuration file) and warn you

## Quick start

In progress

## Advanced features

In progress

## Algorithm output 

In progress

## Creating reports

In progress 

## Report visualisation

In progress 

## Brief architecture description

In progress

## Contributing

Check [contribution guide](contributing.md) for more details.