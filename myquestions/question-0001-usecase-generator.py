import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import random

def generar_caso_de_uso_preparar_datos_robustos():
    n_rows = random.randint(5, 10)
    n_features = random.randint(2, 4)
    
    # Generar datos con algunos valores extremos (outliers)
    data = np.random.randn(n_rows, n_features)
    cols = [f'col_{i}' for i in range(n_features)]
    
    df = pd.DataFrame(data, columns=cols)
    
    # Introducir NaN
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan
    
    target_col = "target"
    df[target_col] = np.random.randint(0, 2, size=n_rows)
    
    input_data = {
        "df": df,
        "target_col": target_col
    }
    
    # --- Lógica de la solución para generar el expected_output ---
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    # Imputación con mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Escalado robusto
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    expected_output = (X_scaled, y)
    
    return input_data, expected_output
