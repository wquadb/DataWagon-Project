
import matplotlib.pyplot as plt
import pandas as pd

def main():

    df = pd.read_csv('dataset/fact_train_test.csv')

    print(df)

    return 0


if __name__ == "__main__":
    main()
