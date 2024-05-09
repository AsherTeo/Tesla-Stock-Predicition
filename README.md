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

In this experiment, I will  explore various window functions including 7, 30, and 60, alongside different learning rates such as 0.01, 0.001, and 0.0001. The goal is to determine the optimal configuration for the task.

The LSTM model comprises 2 layers of LSTM with 50 neurons each, incorporating dropout regularization, followed by one dense layer. Training is performed with a batch size of 8 over 400 epochs.

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/05244cb0-1996-4f9b-9dd4-6cced44a9d9c)


| Window Size | Learning Rate       |    MAE   |   RMSE   | MAPE (%) |
|:-----------:|:-------------:|:---------:|:---------:|:---------:|
|      7      | 0.01   |   5.0380 |   6.8062 | 2.3312 |
|      7      |  0.001  |   5.0064 |   6.8137 | 2.2895 |
|      7      |  0.0001 |   5.5059 |   7.3359 | 2.5535 |
|     30      |  0.01   |   4.9928 |   6.7465 | 2.3012 |
|     30      | 0.001  |   4.9636 |   6.7909 | 2.2870 |
|     30      | 0.0001 |   5.1538 |   6.9946 | 2.3765 |
|     **60**      | **0.01**   |   4.9258 |   6.7241 | **2.2714** |
|     60      | 0.001  |   5.0140 |   6.8050 | 2.3124 |
|     60      | 0.0001 |   5.5312 |   7.3248 | 2.5644 |

The optimal configuration for the LSTM model is a window size of 60, coupled with a learning rate of 0.01, achieving a MAPE of 2.2714%.

## Machine Learning

| Window Size | Model               |    MAE   |   RMSE   | MAPE (%) |
|:-----------:|:-------------------:|---------:|---------:|---------:|
|      7      | Linear Regression   |   4.9843 |   6.7427 |  2.28956 |
|      7      | Gradient Boosting   |   5.7977 |   7.5918 |  2.69867 |
|      7      | Extra Tree          |   5.9703 |   8.0101 |  2.78426 |
|      7      | Random Forest       |   6.0308 |   8.1381 |  2.84164 |



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

## Best Model for each window size

For window size:7

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/c60d05a1-86d9-4a34-8707-a0bb7072a951)


The **ARIMA** model(7,1,0) is best suited for window size 7 with a MAPE of **2.28894** due to its effectiveness in capturing short-term dependencies commonly found in such scenarios. It excels at identifying linear patterns within the data, which is advantageous for short-term forecasting tasks. In contrast, LSTM models may struggle with limited data, leading to potential overfitting. 

For window size:30

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/33f71cfd-4fa3-47b0-bfcd-8b2b4c30e95f)

For a window size of 30, **LSTM** outperforms both ARIMA and Linear Regression with a MAPE of **2.2712**. This suggests that as the window size increases, ARIMA and machine learning approaches begin to perform poorly.

For window size:60 (best overall)

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/766960fb-8e99-4440-8d52-e759d2d34354)

For a window size of 60, **LSTM** outperforms both ARIMA and Linear Regression with an impressive best MAPE score of **2.2692**. This highlights the strength of LSTM in handling longer sequences of data. Meanwhile, ARIMA, Linear Regression, and other machine learning models may struggle with this larger window size, possibly due to overfitting issues. 

## Conclusion

**ARIMA** 
1) Split the data into training and testing sets, with a first 80% for training and remaining  20% for testing.
2) Check if the time series data is stationary by applying the Augmented Dickey-Fuller (ADF) test.
   
   ![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/4d60c9d5-7d7f-49cf-8b64-5c4dd1efb7fa)

   
4) If the p-value obtained from the ADF test is less than 0.05, the data is considered stationary. If not, apply differencing to the data to make it stationary.
5) Repeat steps 2-3 until the time series data becomes stationary.
6) Plot the Partial Autocorrelation Function (PACF) and determine the optimal lag (p).
7) Apply Rolling Forecast by updating the model with testing data and generating forecasts one step ahead iteratively
8) Evaluate the model's performance by comparing the predicted values with the actual test values using metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).
9) Repeat step 5 - 6 for different p-values

**LSTM**
1) Create a window function **k** that transforms a sequence of past data points into features, with the original data as the target variable.
2) Apply MinMaxScaler to normalize the data.
3) Separate the features from the target variable, assigning the features as X and the target as y.
4) Split the scaled data(X and y) into training and testing sets, with a first 80% for training and remaining  20% for testing.
5) Implement a LSTM of 2 LSTM layers of 50 neutron with dropout, one dense layer
6) Configure the batch size to 8 and number of epochs as 400.
7) Perform experiments with different learning rates (0.01, 0.001, and 0.0001) for each **k** value.
8) Evaluate the model's performance by comparing the predicted values with the actual test values using metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).
9) Repeat step 1 - 8 with different **k** values - 7, 30 and 60.

**Machine Learning**
1) Prepare the data by following steps 1 to 4, involving window function creation, normalization, and splitting into training and testing sets.
2) Conduct experiments using various machine learning algorithms such as Linear Regression, Ridge Regression, Gradient Boosting, Random Forest, and others for comparison purposes.
3) Evaluate the performance of each model by comparing the predicted values with the actual test values using metrics including Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).
4) Rank the machine learning algorithms based on their performance in terms of MAPE, sorting them from best to worst.
5) Repeat steps 1 to 4 for different k values: 7, 30, and 60.
