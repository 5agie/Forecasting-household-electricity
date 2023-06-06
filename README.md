# Household Power Consumption Forecasting

This project is an application for forecasting household power consumption. The model uses a dataset containing the household power consumption, then forecasts future consumption using either a univariate or a multivariate model.

## Prerequisites

- Python 3.7+
- Python libraries: pandas, numpy, tensorflow, tkinter, matplotlib, sklearn

## Features

- Load the household power consumption dataset.
- Fill in missing values using mean imputation.
- Resample the dataset per hour.
- Split the data into training and test sets.
- Create and train either a univariate or multivariate model.
- Forecast power consumption using the model.
- Compute and display the Mean Absolute Error (MAE) of the forecast.
- Display the forecast and actual values in a plot.

## Usage

1. Run the script using Python. A graphical user interface will appear.
2. Follow the prompts to select the model type (univariate or multivariate), the number of hours to forecast, and other options.
3. The program will display the MAE of the forecast and a plot comparing the forecast and actual values.

## Note

- This program uses pre-trained weights (files `MultiWeights` and `UniWeights`). Make sure these files are available in the specified locations.
- Make sure all dependencies and the necessary data files are correctly set up in your environment before running the program.
- The file paths in the program should be correctly set according to your directory structure.

## Future Work

Improvements and additional features will be added to the project in the future.