import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

sns.set(style='whitegrid', context='talk')

def plot_sleep_trend(daily_df):
    plt.figure(figsize=(12,5))
    plt.plot(daily_df['date'], daily_df['total_sleep_hours'], marker='o')
    plt.axhline(8, color='green', linestyle='--', label='Recommended 8h')
    plt.title("😴 Total Sleep Duration Over Time")
    plt.xlabel("Date")
    plt.ylabel("Hours Slept")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_weekday_average(daily_df):
    daily_df['weekday'] = pd.to_datetime(daily_df['date']).dt.day_name()
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_avg = daily_df.groupby('weekday')['total_sleep_hours'].mean().reindex(order)
    sns.barplot(x=weekday_avg.index, y=weekday_avg.values, palette='viridis')
    plt.title("🗓️ Average Sleep by Weekday")
    plt.ylabel("Avg Hours")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
