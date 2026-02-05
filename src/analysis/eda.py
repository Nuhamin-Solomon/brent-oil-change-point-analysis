import matplotlib.pyplot as plt
import numpy as np

def plot_price(df):
    plt.figure(figsize=(12,5))
    plt.plot(df["Date"], df["Price"])
    plt.title("Brent Oil Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("USD per Barrel")
    plt.show()

def add_log_returns(df):
    df["log_return"] = np.log(df["Price"]).diff()
    return df
