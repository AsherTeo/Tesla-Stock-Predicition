# Tesla Stock Predicition

## Table of Contents

1) [Overview](#overview)
2) [Objective](#objective)

## Overview

In this project, we analyzed Tesla stock data using the `yfinance` library, spanning from January 1, 2019, to April 29, 2024, and focusing on the 'Close' column. Employing Long Short Term Memory (**LSTM**) with `TensorFlow`, we experimented with various window sizes—7, 30, and 60 days—for forecasting, configuring a batch size of 8, employing two layers with 50 neurons each, and training for 400 epochs.

We also explored the AutoRegressive (AR) component of **ARIMA**, determining the optimal AR component (p) with the `statsmodels` library and PACF Plot. Additionally, we investigated machine learning algorithms like linear regression, Random Forest, and XGBoost, converting the data into a supervised learning format to assess their effectiveness in time series forecasting.

Finally, we compared the results obtained from different algorithms using metrics such as MAE (Mean Absolute Error), RMSE (Root Mean Square Error), and MAPE (Mean Absolute Percentage Error) to assess their performance.

## Objective


## Methodology

**ARIMA** 
1) Check if the graph is stationary by applying adf_test
2) If adf_test is less than 0.05 mean it is stationary if not apply diffencing
3) Repeated


## Result

The table presented below is sorted based on the R2 score of each model.

### 7-Window Size

<details>
  <summary>7-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| **ARIMA(7,1,0)**           |  4.987147 | 6.7420   | **2.28894**   |

### LSTM

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| LR = 0.01              |    5.0593 |  6.8471  |   2.3312  |
| LR = 0.001             |    4.9757 |  6.7710  |   2.2895  |
| LR = 0.0001            |   5.4357  | 7.2619   | 2.5173    |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  MAPE (%) |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  4.9843   | 6.7427   |   2.2895   |
| Gradient Boosting     |  5.7921   |  7.5897   |   2.6958   |
| Extra Tree            |  5.8089   |  7.8397   |   2.7085  |
| Random Forest         |  6.116    |  8.1563   |   2.8719  |
| Ridge Regression      |  6.3457   |  8.2588   |   2.9281   |
| LightGBM (LGM)        |  6.3476   |  8.4223   |   2.9562  |
| CatBoost (CAT)        |  6.3637   |  8.5118   |   2.9580   |
| XGBoost (XGB)         |  6.4597   |  8.5951   |   2.9762   |
| Elastic Net           |  59.559   | 69.913    |   24.856   |

</details>

### 30-Window Size

<details>
  <summary>30-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| ARIMA(30,1,0)          | 	5.2764    |  6.9429   |  2.4302  |

### LSTM

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| **LR = 0.01**              |   4.94155 | 6.7306   |   **2.2712**  |
| LR = 0.001             |   4.9847  | 6.7892  | 	2.2907  |       
| LR = 0.0001            |   5.1551|  |  6.9886 |2.3981 |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  MAPE (%) |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  5.2877  | 6.9560    |   2.4366  |
| Gradient Boosting     |  5.7181  |  7.6020  |  2.6707   |
| Extra Tree            |  5.8089  |  8.0917  |   2.7830  |
| Random Forest         |  6.0941  |  7.9411    |   2.8527  |
| LightGBM (LGM)        |  6.2800  |  8.2970  |   2.9040  |
| Ridge Regression      |  6.3593  | 8.2356    |   2.9360  |
| XGBoost (XGB)         |  7.0187  |  9.1641  |  3.2768 |
| CatBoost (CAT)        |  7.9380  |  10.239 |   3.7377  |
| Elastic Net           | 56.7614   | 67.2448   |   	23.6125  |

</details>

### 60-Window Size

<details>
  <summary>60-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| ARIMA(60,1,0)           | 5.4316	   | 7.1240 | 2.5042 |


### LSTM

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| **LR = 0.01**              |    4.948	 |  6.7376    |  **2.2692** |
| LR = 0.001             |    4.955  |  6.770 	|  2.2849 |
| LR = 0.0001            |  5.2721   |  7.075 |  2.444  |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  MAPE (%) |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  5.5622  |  7.2369  |   2.5704  |
| Gradient Boosting     |  5.6747  |  7.4616  |   2.6456  |
| Random Forest         |  5.7565  |  7.6322  |   2.6943  |
| Extra Tree            |  6.111  |  7.9947   |   2.8482  |
| Ridge Regression      |  6.3595  |  8.1879  |   2.9454  |
| LightGBM (LGM)        |  6.4789  |  8.6404  |   2.9934  |
| XGBoost (XGB)         |  6.6262  |  8.9633  |   3.0705  |
| CatBoost (CAT)        |  8.7131  | 11.0934  |   4.0871  |
| Elastic Net           | 53.1747   |  63.679  |   22.055  |

</details>
