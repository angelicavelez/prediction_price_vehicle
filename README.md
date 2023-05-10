# Predicción del precio de un vehículo 

Crear un modelo de machine learning que determine rápidamente valor de mercado de vehículos en venta para la compañía *Rusty Bargain* que vende vehículos usados, la cual está desarrollando una aplicación para atraer nuevos clientes.

EL objetivo de nuestro modelo predictivo, es evaluar nuestro modelo en base a:

- la **calidad** de la predicción
- la **velocidad** de la predicción y **tiempos** de entrenamiento.

Aplicamos técnicas de potenciación de gradiente y ajustes de hiperparámetros en nuestros modelos de machine learning.

# Bibliotecas usadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor