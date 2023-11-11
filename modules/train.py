import torch
import scipy
import pandas as pd
import support_funcs as sup

def main_show():
    
    df = pd.read_csv('dataset/fact_train_test.csv')
    
    df, ds = sup.preprocess(df)

    sup.show_timeseries(ds)
    
    sup.show_correlation(df['real_weight'], df['real_wagon_count'])

if __name__ == "__main__":

    main_show()


    
