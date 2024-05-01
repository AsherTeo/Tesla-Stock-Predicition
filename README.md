# Tesla Stock Predicition

## Result

The table presented below is sorted based on the R2 score of each model.

### 7-Window Size

<details>
  <summary>7-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| ARIMA(7,1,0)           |  4.9590	 | 6.6802   | 0.96847   |

### LSTM

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| LR = 0.01              |    4.9677	 |  6.6977    |   0.96831  |
| LR = 0.001             |            | | 	|       |
| LR = 0.0001            |   |  |   |

### Machine Learning Models

| Model                 |   MAE    |   RMSE   |  R2 Score |
|-----------------------|----------|----------|-----------|
| Linear Regression     |  4.9843   | 6.7427   |   0.96788   |
| Ridge Regression      |  5.0999   |  6.8881   |   0.9664   |
| Gradient Boosting     |  5.8007   |  7.5935   |   0.9592   |
| LightGBM (LGM)        |  5.9469   |  7.8029   |   0.9569  |
| Extra Tree            |  5.9567   |  7.9926   |   0.9548  |
| Random Forest         |  5.9315   |  8.0511   |   0.9542  |
| CatBoost (CAT)        |  6.2924   |  8.4183   |   0.9499   |
| XGBoost (XGB)         |  6.3653   |  8.5146   |   0.9487   |
| Elastic Net           |  28.556   | 33.950    |   0.1858   |

</details>

### 30-Window Size

<details>
  <summary>30-Window Size</summary>
  
### ARIMA

| Model                  |    MAE    |   RMSE   |  R2 Score | 
|------------------------|-----------|----------|-----------|
| ARIMA(30,1,0)          | 	5.263    |  6.9429   |  0.96595  |

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
