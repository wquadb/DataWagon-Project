import torch
import scipy
import pandas as pd
import support_funcs
import support_funcs as sup

if __name__ == "__main__":
    
    df = pd.read_csv('dataset/fact_train_test.csv')
    
    df = preprocess(df)

    sup.show_timeseries(df)
    
    print(df)
