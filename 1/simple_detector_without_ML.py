import pandas as pd
import matplotlib.pyplot as plt

def main():
    filename = "telemetry_dzz_sem1.csv"

    df = pd.read_csv(filename, parse_dates=['timestamp'])
    print(df.head())
    print("Размер выборки:", len(df))
    print(df.describe())
    print(df)
    
    print()

    ax = df.plot(x='temperature', y='timestamp', kind='scatter')
    plt.show()

if __name__ == "__main__":
    main()

