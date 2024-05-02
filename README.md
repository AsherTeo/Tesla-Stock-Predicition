# Tesla Stock Predicition

## Result

The table presented below is sorted based on the R2 score of each model.

### 7-Window Size

<details>
  <summary>7-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| ARIMA(7,1,0)           |  4.987147 | 6.7420   | 2.28894   |

### LSTM

| Model                  |    MAE    |   RMSE   |  MAPE (%) | 
|------------------------|-----------|----------|-----------|
| LR = 0.01              |    4.97083	 |  6.7430    |   2.2724  |
| LR = 0.001             |            | | 	|       |
| LR = 0.0001            |   |  |   |

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
| LR = 0.01              |   4.94155 | 6.7306   |   2.2712  |
| LR = 0.001             |   4.9847  | 6.7892  | 	2.2907  |       
| LR = 0.0001            |   |  |   |

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

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| ARIMA(60,1,0)           | 5.4316	   | 7.1240    | 0.9641 |


### LSTM

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| LR = 0.01              |    4.924	 |  6.67    |   0.9685  |
| LR = 0.001             |            | | 	|       |
| LR = 0.0001            |   |  |   |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  R2 Score |
|-----------------------|----------|----------|-----------|
| Ridge Regression      |  5.4760  |  7.1571  |   0.9638  |
| Linear Regression     |  5.5622  |  7.2369  |   0.9630  |
| Gradient Boosting     |  5.6785  |  7.4631  |   0.9606  |
| Random Forest         |  5.6631  |  7.5703  |   0.9595  |
| Extra Tree            |  5.7269  |  7.626   |   0.9589  |
| LightGBM (LGM)        |  6.2235  |  8.2777  |   0.9516  |
| XGBoost (XGB)         |  6.5618  |  8.9237  |   0.9437  |
| CatBoost (CAT)        |  8.7157  | 11.0949  |   0.9130  |
| Elastic Net           | 25.415   |  31.025  |   0.3200  |

</details>
