# Tesla Stock Predicition

## Table of Contents

1) [Overview](#overview)
2) [Objective](#objective)

## Overview

In this project, I utilized the `yfinance` library to extract Tesla stock data and experimented with various prediction methods, including LSTM, ARIMA, and machine learning algorithms like linear regression. 

LSTM, known for its proficiency in analyzing long sequences of data, was tested for its efficacy in long-term forecasting using `tensorflow` library. 
In contrast, ARIMA utilizes the PACF plot to determine the lag for the autoregressive component (p) is typically small. Additionally, machine learning techniques such as linear regression were explored to assess their suitability for shorter window predictions. The objective was to determine whether LSTM's longer window or ARIMA's shorter window, or perhaps machine learning algorithms, would yield superior predictions for Tesla's stock

## Objective

1) 

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
| Linear Regression     | 4.984  | 6.7427    |   2.2895  |
| Gradient Boosting     |  5.7921  |  7.5897  |  2.6958   |
| Extra Tree            |  5.8089  |  7.8397  |   2.7085  |
| Random Forest         |  6.1163  |  8.1563    |   2.8719  |
| Ridge Regression      |  6.3457  | 8.2588    |   2.9281  |
| LightGBM (LGM)        |  6.3476  |  8.4223  |   2.9562  |
| CatBoost (CAT)        |  6.3637  |  8.5118 |   2.9580  |
| XGBoost (XGB)         |  6.4597  |  8.5951  |  2.9762 |
| Elastic Net           | 59.5591   | 69.9136   |   	24.856  |

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
