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
