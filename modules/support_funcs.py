from matplotlib import pyplot as plt
import pandas as pd


def show_correlation(series_1: pd.Series, series_2: pd.Series):
    
    """
    takes DataFrame and two columns (float/int)
    then shows correlation on x, y surface
    """
    
    print(f"Correlation between {series_1.name} and {series_2.name}:\n")
    print(series_1.corr(series_2), '\n')

    x = series_1
    y = series_2

    fig, ax =  plt.subplots(figsize=(9, 7), dpi=100)

    ax.scatter(x, y, alpha=0.3, s=10)

    plt.xlabel(f"{series_1.name}", fontsize=14, labelpad=15)
    plt.ylabel(f"{series_2.name}", fontsize=14, labelpad=15)
    plt.show()

    return 0

def show_timeseries(df):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25,15))
    df.plot(kind='line', x="rps", y='day', color='b', ax = axes, rot=0)
    plt.show()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    """
    takes DataFrame with column 'date' and change dates 
    to the number of days from the earliest day

    also creates duplicate for visualising graph
    """

    df['period'] = pd.to_datetime(df['period'], format=r'%Y-%m-%d')

    ds = df.pivot_table(columns=['period'], aggfunc='size')

    return df, ds

if __name__ == "__main__":

    pass
