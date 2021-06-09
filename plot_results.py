from argparse import ArgumentParser
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def main(args):
    data = pd.read_csv(args.csv_path)

    x_data = data["pages"].values.reshape((-1, 1))

    model = LinearRegression()
    model.fit(x_data, data["time"])

    print(f"fork with no maps: {model.intercept_:.2f} us")

    example_pages = 1000
    print(f"cost of {example_pages} pages: {model.coef_[0] * example_pages:.2f} us")

    fig, ax = plt.subplots()
    ax.scatter(x_data, data["time"])
    ax.plot(x_data, model.predict(x_data))
    ax.set_xlabel("RSS (pages)")
    ax.set_ylabel("Time (us)")

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    main(args)