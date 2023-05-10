# Predicción del precio de un vehículo 

Crear un modelo de machine learning que determine rápidamente valor de mercado de vehículos en venta para la compañía *Rusty Bargain* que vende vehículos usados, la cual está desarrollando una aplicación para atraer nuevos clientes.

EL objetivo de nuestro modelo predictivo, es evaluar nuestro modelo en base a:

- la **calidad** de la predicción
- la **velocidad** de la predicción y **tiempos** de entrenamiento.

Aplicamos técnicas de potenciación de gradiente y ajustes de hiperparámetros en nuestros modelos de machine learning.

# Bibliotecas usadas
numpy, pandas, matplotlib.pyplot, time, sklearn.model_selection import train_test_split, GridSearchCV, 
sklearn.preprocessing import StandardScaler, sklearn.linear_model import LinearRegression, sklearn.tree import DecisionTreeRegressor, 
sklearn.ensemble import RandomForestRegressor, sklearn.metrics import mean_squared_error, sklearn.model_selection import cross_val_score,
xgboost import XGBRegressor, lightgbm import LGBMRegressor, catboost import CatBoostRegressor