import pandas as pd

def detectar_outliers_iqr(X):
    df_copy = X.copy()
    col = 'valores'
    
    Q1 = df_copy[col].quantile(0.25)
    Q3 = df_copy[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_copy["es_outlier"] = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
    
    return df_copy
