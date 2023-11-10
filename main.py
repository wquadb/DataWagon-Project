import support_func
import matplotlib.pyplot as plt
import pandas as pd

def main():

    df = pd.read_csv(input())

    print(df.head())

    return 0


if __name__ == "__main__":
    main()
