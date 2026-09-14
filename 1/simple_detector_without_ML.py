import pandas as pd
import matplotlib.pyplot as plt

def get_intervals(df, col='anomaly'):
    mask = df[col] == 1
    grp = (mask != mask.shift()).cumsum()
    
    intervals = (
        df[mask]
        .groupby(grp)
        .agg(
            start=('timestamp', 'first'),
            end=('timestamp', 'last'),
            rows=('timestamp', 'size')
        )
        .reset_index(drop=True)
    )
    intervals['duration_min'] = (intervals['end'] - intervals['start']).dt.total_seconds() / 60
    return intervals

def main():
    filename = "telemetry_dzz_sem1.csv"

    df = pd.read_csv(filename, parse_dates=['timestamp'])
    print('TELEMETRY (5)')
    print(df.head())

    print()
    print("Размер выборки:", len(df))
    print(df.describe())
    # print(df)
    
    print()

    # temp = df.plot(x='timestamp', y='temperature', kind='scatter')
    # vol = df.plot(x='timestamp', y='voltage', kind='scatter')
    # cur = df.plot(x='timestamp', y='current', kind='scatter')
    # ang = df.plot(x='timestamp', y='angular_velocity', kind='scatter')
    # plt.show()

    df['anomaly'] = 0
    df.loc[df['temperature'] > 45, 'anomaly'] = 1
    df.loc[df['temperature'] < 20, 'anomaly'] = 1
    df.loc[df['voltage'] > 29, 'anomaly'] = 1
    df.loc[df['voltage'] < 27, 'anomaly'] = 1
    df.loc[df['current'] > 7, 'anomaly'] = 1
    df.loc[df['current'] < 0, 'anomaly'] = 1
    df.loc[df['angular_velocity'] > 0.2, 'anomaly'] = 1
    df.loc[df['angular_velocity'] < -0.1, 'anomaly'] = 1

    print('WITH ANOMALY')
    print(df.head())

    print()
    print("INTERVALS")
    intervals = get_intervals(df)
    print(intervals)

    params = ['temperature', 'voltage', 'current', 'angular_velocity']

    for col in params:
        plt.figure(figsize=(12, 4))
        plt.plot(df['timestamp'], df[col], label=col, color='tab:blue')
        plt.scatter(
            df.loc[df['anomaly'] == 1, 'timestamp'],
            df.loc[df['anomaly'] == 1, col],
            color='red', s=25, label='anomaly', zorder=5
        )
        for _, row in intervals.iterrows():
            plt.axvspan(row['start'], row['end'], color='red', alpha=0.2)
        plt.title(f'Telemetry: {col}')
        plt.xlabel('timestamp')
        plt.ylabel(col)

    print()
    print('CLOSE FIGURES')
    print()

    plt.show()


if __name__ == "__main__":
    main()

