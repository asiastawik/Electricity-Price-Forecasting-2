# Electricity Price Forecasting with AR(7) and ARX(7) Models

## Project Overview
This project involves analyzing electricity price data from the GEFCom and Nord Pool datasets (2013-2016). Using Python libraries such as `pandas` and `numpy`, we will implement AR(7) and ARX(7) forecasting models, evaluate their performance, and visualize the results.

## Task 1: Data Preparation and Forecasting

### 1.1: Downloading the Dataset
- Download and unpack the datasets from the Electricity price data (GEFCom, Nord Pool 2013-2016) available in ePortal under T03+L03 The forecaster’s toolbox.

### 1.2: Loading the Data
- Load the contents of the `GEFCOM.txt` file using either `numpy` or `pandas`.
- Ensure to examine the columns as described on ePortal.

### 1.3: Preparing Forecasts
- **Forecast Preparation**: Create new columns with forecasts spanning from January 1, 2013, until the end of the dataset, using the following models:
  
  1. **AR(7) Single**:
     - Estimate this model only once, separately for each hour of the day, using data before the end of 2012 for training.
  
  2. **AR(7) Rolling**:
     - Re-estimate daily using a rolling calibration window, separately for each hour of the day.
  
  3. **ARX(7) Rolling**:
     - Re-estimate daily using a rolling calibration window, separately for each hour of the day.

### 1.4: Performance Evaluation
- **Error Metrics**: For each model, compute the following metrics:
  1. **Overall Performance**:
     - Report the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for the whole out-of-sample test period (print the results to the console).
  
  2. **Weekly Performance**:
     - Calculate the MAE and RMSE for each full week (Monday-Sunday, ignoring partial weeks).
     - Prepare a bar chart of the errors for consecutive weeks.
     - For each forecast, print the weeks (begin dates) with the lowest and highest scores (for each metric) along with the corresponding errors.
  
### 1.5: Visualization
- Plot the zonal price and its ARX(7) rolling forecast for all data from November and December 2013. 
- Ensure the x-ticks are set correctly, either by manual setting (e.g., one tick every week) or by parsing the dates. The x-tick labels should follow the format DD_MM_YYYY (e.g., 20_10_2013).
