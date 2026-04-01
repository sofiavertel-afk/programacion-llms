import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import random

def generar_caso_de_uso_preparar_datos():

    n_rows = random.randint(5, 10)
    n_features = random.randint(2, 4)

    data = np.random.randn(n_rows, n_features)
    cols = [f'col_{i}' for i in range(n_features)]

    df = pd.DataFrame(data, columns=cols)

    # introducir NaN
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan

    target_col = "target"
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    output_data = (X, y)

    return input_data, output_data
