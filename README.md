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
| LR = 0.01              |    5.0363	 |  6.7515    |   0.9677  |
| LR = 0.001             |            | | 	|       |
| LR = 0.0001            |   |  |   |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  R2 Score |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  4.9816   |  6.7375   |   0.9679   |
| Ridge Regression      |  5.0865   |  6.8715   |   0.9666   |
| Gradient Boosting     |  5.7893   |  7.6188   |   0.9590   |
| Extra Tree            |  5.9610   |  7.9926   |   0.9548  |
| LightGBM (LGM)        |  6.1465   |  8.0279   |   0.9544  |
| Random Forest         |  6.0442   |  8.0498   |   0.9542  |
| CatBoost (CAT)        |  6.4080   |  8.4544   |   0.9495   |
| XGBoost (XGB)         |  6.3797   |  8.5125   |   0.9488   |
| Elastic Net           | 28.3497   | 33.791    |   0.1935   |

</details>

### 30-Window Size

<details>
  <summary>30-Window Size</summary>
  
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
  <summary>60-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| ARIMA(60,1,0)           | 5.4316	   | 7.1240    | 0.9641 |
| ARIMA(60,1,7)           ||  | 	|       |
| ARIMA(60,1,8)           |  |  |   |

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
