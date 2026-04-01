import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import random

def generar_caso_de_uso_entrenar_modelo():

    n_rows = random.randint(10, 20)
    n_cols = random.randint(2, 5)

    data = np.random.randn(n_rows, n_cols)
    cols = [f'col_{i}' for i in range(n_cols)]

    df = pd.DataFrame(data, columns=cols)

    target_col = "target"
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = LogisticRegression()
    model.fit(X, y)

    output_data = model

    return input_data, output_data
