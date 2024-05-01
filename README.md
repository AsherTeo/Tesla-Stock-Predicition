# Tesla Stock Predicition

## Result

The table presented below is sorted based on the R2 score of each model.

### 8-Window Size

<details>
  <summary>8-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| ARIMA(8,1,0)           | 5.003	   | 6.758    | 0.9676 |
| ARIMA(8,1,7)           ||  | 	|       |
| ARIMA(8,1,8)           |  |  |   |

### LSTM

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| LR = 0.01              |    4.924	 |  6.67    |   0.9685  |
| LR = 0.001             |            | | 	|       |
| LR = 0.0001            |   |  |   |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  R2 Score |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  4.981   |  6.737   |   0.967   |
| Ridge Regression      |  5.086   |  6.871   |   0.966   |
| Gradient Boosting     |  5.789   |  7.618   |   0.959   |
| Extra Tree            |  5.961   |  7.992   |   0.9548  |
| LightGBM (LGM)        |  6.146   |  8.027   |   0.9544  |
| Random Forest         |  6.044   |  8.049   |   0.9542  |
| CatBoost (CAT)        |  6.408   |  8.454   |   0.949   |
| XGBoost (XGB)         |  6.379   |  8.512   |   0.948   |
| Elastic Net           | 28.3497  | 33.791   |   0.193   |

</details>

### 30-Window Size

<details>
  <summary>8-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| ARIMA(30,1,0)          | 	5.263    |  6.939   |  0.9658   |
| ARIMA(30,1,7)           ||  | 	|       |
| ARIMA(30,1,8)           |  |  |   |

### LSTM

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| LR = 0.01              |   5.0165	 |  6.7490    |   0.9679  |
| LR = 0.001             |            | | 	|       |
| LR = 0.0001            |   |  |   |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  R2 Score |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  5.2877  | 6.956    |   0.9658  |
| Ridge Regression      |  5.2583  | 6.999    |   0.9653  |
| Gradient Boosting     |  5.7199  |  7.5811  |  0.9594   |
| Random Forest         |  5.9181  |  7.9043  |   0.9558  |
| LightGBM (LGM)        |  6.1583  |  8.2863  |   0.9515  |
| Extra Tree            |  6.2216  |  8.3228  |   0.9510  |
| XGBoost (XGB)         |  6.9564  |  9.0849  |  0.9417   |
| CatBoost (CAT)        |  7.9499  |  10.2568 |   0.9256  |
| Elastic Net           | 26.912   | 32.614   |   0.2487  |

</details>

### 60-Window Size

<details>
  <summary>8-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| ARIMA(8,1,0)           | 5.003	   | 6.758    | 0.9676 |
| ARIMA(8,1,7)           ||  | 	|       |
| ARIMA(8,1,8)           |  |  |   |

### LSTM

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| LR = 0.01              |    4.924	 |  6.67    |   0.9685  |
| LR = 0.001             |            | | 	|       |
| LR = 0.0001            |   |  |   |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  R2 Score |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  4.981   |  6.737   |   0.967   |
| Ridge Regression      |  5.086   |  6.871   |   0.966   |
| Gradient Boosting     |  5.789   |  7.618   |   0.959   |
| Extra Tree            |  5.961   |  7.992   |   0.9548  |
| LightGBM (LGM)        |  6.146   |  8.027   |   0.9544  |
| Random Forest         |  6.044   |  8.049   |   0.9542  |
| CatBoost (CAT)        |  6.408   |  8.454   |   0.949   |
| XGBoost (XGB)         |  6.379   |  8.512   |   0.948   |
| Elastic Net           | 28.3497  | 33.791   |   0.193   |

</details>
