import sys

import matplotlib.pyplot as plt
import pandas


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["step", "batch_size", "count", "t"])


def main(statsfile, bs):
    data = parse_stats_file(statsfile)

    fig, ax = plt.subplots()

    for count in range(1, 11):
        data.where((data["batch_size"] == bs) & (data["count"] == count)).plot(
            x="step", y="t", ax=ax, label=f"count={count}"
        )

    bs = bs if bs > 1 else "off"
    # adding title to the plot
    plt.title(f"Time per element (batch={bs})")

    # adding Label to the x-axis
    plt.ylim(top=1000, bottom=0)
    plt.xlabel("step")
    plt.ylabel("us")
    # adding legend to the curve
    plt.legend()

    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    bs = int(sys.argv[2])
    main(filename, bs)
