# Tesla Stock Predicition

## Table of Contents

1) [Overview](#overview)
2) [Technical Analysis ](#technical-analysis)
3) [ARIMA](#ARIMA)
4) [LSTM](#LSTM)
5) [Transformer](#Transformer)
6) [Machine Learning](#machine-learning)
7) [Conclusion](#Conclusion)
8) [Installation](#Installation)
   
## Overview

In this project, we analyzed Tesla stock data using the `yfinance` library, spanning from January 1, 2019, to April 29, 2024, and focusing on the 'Close' column. The initial segment covered the **Technical Analysis**, wherein we explored indicators such as MACD (Moving Average Convergence Divergence), Relative Strength Index (RSI), and bearish and bullish divergences. These tools are instrumental in identifying trends and patterns, aiding in informed buy and sell decisions to follow market movements.

The subsequent part focusing on **Time Series**, where we employed various time series models including ARIMA, LSTM, Transformer and Machine learning techniques. For Long Short Term Memory (**LSTM**) with `TensorFlow`, we experimented with various window sizes—7, 30, and 60 days—for forecasting, configuring a batch size of 32, employing two layers with 50 neurons each, and training for 400 epochs.  Additionally, we explored the **Transformer** model, given its prominence in the field.

Moreover, we explored the AutoRegressive (AR) component of **ARIMA**, determining the optimal AR component (p) with the `statsmodels` library and PACF Plot. Further, we investigated machine learning algorithms such as linear regression, Random Forest, and XGBoost, transforming the data into a supervised learning format to evaluate their efficacy in time series forecasting.

Finally, we compared the results obtained from different algorithms using metrics such as MAE (Mean Absolute Error), RMSE (Root Mean Square Error), and MAPE (Mean Absolute Percentage Error) to assess their performance.

## Technical Analysis 

Technical Analysis is a method used by traders and investors to forecast the future direction of the financial markets. Some of the popular technical analysis tools are MACD (Moving Average Convergence Divergence), Bollinger Bands , Relative Strength Index (RSI), and etc. 

Relative Strength Index (RSI) is the most popular technical indicator to gauage the momentum  of the stock's price movements to determine whether it's currently in an overbought or oversold condition. Normally, when the RSI value reach above 70, it suggests that the stock may be overbought which highly indicate that the stock price will reversal in the near future. Conversely, when the RSI value descends below 30, it signals potential oversold conditions, implying an impending rise in stock price. 

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/23d6fecb-ef3c-41d1-b68a-c8c2040859ff)

In addition to these overbought and oversold conditions, traders also closely monitor **bearish and bullish divergences** in the Relative Strength Index (RSI).

For my upcoming update, I intend to develop a simulator that calculates profit based on the invested amount, taking into account bearish and bullish divergences in the RSI.

## Time Series

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

The LSTM model comprises 2 layers of LSTM with 50 neurons each, incorporating dropout regularization, followed by one dense layer. Training is performed with a batch size of 32 over 400 epochs.

![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/e2a2ba56-f509-47b5-8cf6-a1e24e645e5f)


| Window Size | Learning Rate       |    MAE   |   RMSE   | MAPE (%) |
|:-----------:|:-------------:|:---------:|:---------:|:---------:|
|      7      | 0.010645732663668563   |   4.9325 |   6.7327 | 2.2656 |
|      **30**      |  0.010407419281699032  |   4.9169 |   6.7028 | **2.2587** |
|      60      |  0.009866144058738683 |   4.9386 |   6.6488 | 2.2669 |


The optimal configuration for the LSTM model is a window size of 30, coupled with a learning rate of 0.010407419281699032, achieving a MAPE of **2.2587%**.

## Transformer

Transformers are popular deep learning model known for their capability to handle sequential data by using self-attention mechanisms, which not only dominated in NLP but also have proven their versatility across various domains, including computer vision tasks. Our experiment aims to explore the efficacy of transformers in handling time series tasks, leveraging their self-attention mechanisms and adaptability to sequential data using different window size. 

We will be ulitizing `optuna` on some of the parameters such as learning rate, number of heads, head size and number of feed forward dimensions. 


![image](https://github.com/AsherTeo/Tesla-Stock-Predicition/assets/78581569/6df1a967-adb1-445e-9a3c-fc0dd39d42b5)

| Window Size | Learning Rate       | No. of head   | Head Size | FFD |    MAE   |   RMSE   | MAPE (%) |
|:-----------:|:-------------------:|:-------------:|:---------:|:---:|:--------:|:--------:|:--------:|
|      **7**      | 0.0014332163970285864  | 8 | 512 | 4 |4.9899 | 6.7673 | **2.2982** |
|      30     | 0.0019745514796305342  | 16 | 256 | 8 |5.0728 | 6.9158 | 2.3362 |
|      60     | 0.0019722952704556334  | 8 | 256|2 |5.4437 | 7.1714 | 2.5046 |


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

## Conclusion

In conclusion, the **LSTM** model with a window size of **30** achieved the best performance with a MAPE of **2.2587%**, while the Transformer model yielded the worst result with a MAPE of 2.2982%. Time series data is inherently sequential, with each data point reliant on its preceding ones. LSTM networks are purpose-built for handling such sequential data, enabling them to effectively capture temporal dependencies. However, Transformers process input data in parallel, which may not fully exploit the sequential nature of time series data. On the other hand, ARIMA (7,1,0)  with rolling forecasting and Linear Regression showed promising results with a MAPE of 2.28894% and 2.28956% respectively.

Ultimately, ARIMA and Linear Regression are suitable for short window forecasting but less effective for longer windows, and they train quickly. In contrast, LSTM is the ideal model with good MAPE for both long and short windows, although it requires fine-tuning and longer training times. As for the Transformer model, it proves less than ideal for time series forecasting due to longer training times and average results. However, advancements like Autoformer, which incorporates Autoregressive components as position encoding to capture temporal information may show promise for improving Transformer-based time series forecasting.


 ## Installation
 
The code is developed using Python version 3.10.14  If Python is not already installed on your system, you can download it [here](https://www.python.org/downloads/). If your current Python version is lower than 3.10.14  you can upgrade it using the pip package manager. Make sure you have the latest version of pip installed. To install the necessary packages and libraries, execute the following command in the project directory after cloning the repository:

```bash
pip install -r requirements.txt
```
For window GPU installation, make sure python version is 3.7 - 3.10, tensorflow version < 2.11, cuda tool kit = 11.2 and cudnn = 8.1.0

This is an example to create a new environment for the following requirement:

```bash
conda create --name yourenv python=3.10
conda activate yourenv
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10 
```
You can check your cuda version using this command in terminal

```bash
nvcc --version
```
