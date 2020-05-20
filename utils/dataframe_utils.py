import pandas as pd

def get_column_categories(df,verbose = 1):
    if verbose > 0: print('Column Values:')
    cols = df.columns
    uniques = {}
    for col in cols:
        if verbose > 0: print(col,':',df[col].unique())
        uniques.update({col:df[col].unique()})
    return uniques