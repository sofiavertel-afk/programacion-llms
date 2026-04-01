import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_estadisticas():

    n_rows = random.randint(5, 10)
    n_cols = random.randint(2, 5)

    data = np.random.randn(n_rows, n_cols)
    cols = [f'col_{i}' for i in range(n_cols)]

    df = pd.DataFrame(data, columns=cols)

    input_data = {
        "df": df.copy()
    }

    medias = df.mean().to_numpy()
    stds = df.std().to_numpy()

    output_data = (medias, stds)

    return input_data, output_data
