# Tesla Stock Predicition

## Table of Contents

1) [Overview](#overview)
2) [Objective](#objective)

## Overview

In this project, we analyzed Tesla stock data using the `yfinance` library, spanning from January 1, 2019, to April 29, 2024, and focusing on the 'Close' column. Employing Long Short Term Memory (**LSTM**) with `TensorFlow`, we experimented with various window sizes—7, 30, and 60 days—for forecasting, configuring a batch size of 8, employing two layers with 50 neurons each, and training for 400 epochs.

We also explored the AutoRegressive (AR) component of **ARIMA**, determining the optimal AR component (p) with the `statsmodels` library and PACF Plot. Additionally, we investigated machine learning algorithms like linear regression, Random Forest, and XGBoost, converting the data into a supervised learning format to assess their effectiveness in time series forecasting.

Finally, we compared the results obtained from different algorithms using metrics such as MAE (Mean Absolute Error), RMSE (Root Mean Square Error), and MAPE (Mean Absolute Percentage Error) to assess their performance.

## Objective

The objective of this project is to determine the optimal window size for forecasting using ARIMA, LSTM, and machine learning algorithms. By experimenting with different window sizes, ranging from shorter to longer durations, the project aims to identify the most effective approach for predicting Tesla stock prices. This involves exploring how each method performs with various window sizes and comparing their accuracy in forecasting. Ultimately, the project seeks to understand whether shorter or longer window sizes are more suitable for each forecasting technique.

## ARIMA

ARIMA, a time series forecasting technique, integrates autoregressive (AR) and moving average (MA) components. The AR component employs past observations of a variable to predict its current value, while the MA component focuses on the relationship between the current value of a series and past prediction errors.

Stock prices are often non-stationarity due to trends and seasonality. Therefore, autocorrelation function (ACF) and partial autocorrelation function (PACF) plots are used to analyze these patterns and characteristics.
![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/e4e4a769-3a22-4775-bfc5-badaf4028189)

We can observe that most of the lagged values in the ACF plots are close to 1, suggesting a strong autocorrelation, which is indicative of non-stationarity. Another powerful method to determine stationarity is by applying the Augmented Dickey-Fuller (ADF) test. ADF testing helps ascertain whether differencing is necessary to achieve stationarity in the data.

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/51ed2387-997d-4163-be0d-d7ebdbad08e7)

After first differencing the time series, we can observe that the mean and varience tend to be more stable over time compared to the original series.

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/55930bd1-438d-426f-b7b9-38851d5df19c)

To determine the p components, we examine the PACF plot, where we observe four lagged values above the significance line at lags 7, 9, 18, and 24. Since I havee decided not to include the MA component, I'll focus on experimenting with the AR component using these specific lagged values (7, 9, 18, and 24).

I splited the dataset into two parts, using the first 80% for training and remaining 20% for testing. 

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/3613a75d-bacb-4229-b8b5-a8633d9aedd4)

Before applying rolling forecasts, ARIMA models may struggle to accurately predict distant future data points, leading to uncertainty. A beneficial approach is to implement rolling forecasts. In rolling forecasts, the ARIMA model is continually updated using testing data to predict the next data point. By iteratively refining the model with newly observed data, rolling forecasts can provide more reliable and adaptive predictions. 

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/e2c4d525-8cfa-43a0-b97f-3034c5f13606)

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/5b66ab7f-7888-45c4-b684-6982b10a5c4c)

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|:------------------------:|:-----------:|:----------:|:-----------:|
| **ARIMA(7,1,0)**       |  4.98714 | 6.7420   | **2.28894**   |
| ARIMA(9,1,0)       |  5.03233 | 6.7649   | 2.31162  |
| ARIMA(18,1,0)      |  5.13619 | 6.8585   | 2.35835   |
| ARIMA(24,1,0)     |  5.23434 | 6.9381   | 2.40312   |

ARIMA(7,1,0) is the optimal ARIMA model.

## LSTM

Long Short-Term Memory (LSTM) is a popular time series forecasting model for processing  long sequence of data due to their unique architecture. In short, LSTMs utilize memory cells and gating mechanisms to manage information flow, retaining essential data over time while filtering out irrelevant information. These features enable LSTMs to effectively capture long-term dependencies in data sequences, addressing issues like the vanishing gradient problem. 

In this experiment, I will  explore various window functions including 7, 30, and 60, alongside different learning rates determined by `optuna`. The goal is to determine the optimal configuration for the task.

The LSTM model comprises 2 layers of LSTM with 50 neurons each, incorporating dropout regularization, followed by one dense layer. Training is performed with a batch size of 8 over 400 epochs.

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/e2a2ba56-f509-47b5-8cf6-a1e24e645e5f)


| Window Size | Learning Rate       |    MAE   |   RMSE   | MAPE (%) |
|:-----------:|:-------------:|:---------:|:---------:|:---------:|
|      7      | 0.010645732663668563   |   4.9325 |   6.7327 | 2.2656 |
|      **30**      |  0.010407419281699032  |   4.9169 |   6.7028 | **2.2587** |
|      60      |  0.009866144058738683 |   4.9386 |   6.6488 | 2.2669 |


The optimal configuration for the LSTM model is a window size of 30, coupled with a learning rate of 0.010407419281699032, achieving a MAPE of **2.2587%**.

## Transformer


## Machine Learning

In my final experiment, I will explore various machine learning algorithms including Linear Regression, Gradient Boosting, Extra Trees, XGBoost, etc. I will also investigate different window sizes similar to those used in the LSTM model. 

After evaluation, it was found that Linear Regression achieved the lowest MAPE of **2.28956%**, making it the best model compared to the others.

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/957ceda5-323a-4fa4-8d81-b3b1951c207b)


| Window Size | Model               |    MAE   |   RMSE   | MAPE (%) |
|:-----------:|:-------------------:|:---------:|:---------:|:---------:|
|      **7**      | **Linear Regression**   |   4.9843 |   6.7427 |  **2.28956** |
|      7      | Gradient Boosting   |   5.7977 |   7.5918 |  2.69867 |
|      7      | Extra Tree          |   5.9703 |   8.0101 |  2.78426 |
|      30      | Linear Regression   |   5.2877 |   6.9560 |  2.4366 |
|      30      | Gradient Boosting   |   5.7207 |   7.5830 |  2.67155 |
|      30      | Extra Tree          |   6.0142 |   8.0839 |  2.7890 |
|      60      | Linear Regression   |   5.5622 |   7.2369 |  2.57045 |
|      60      | Gradient Boosting   |   5.6854 |   7.4822 |  2.65154 |
|      60      | Random Forest      |   5.72054 |   7.6123 |  2.67572 |


