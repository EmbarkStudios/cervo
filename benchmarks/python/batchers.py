import sys

import matplotlib.pyplot as plt
import pandas


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "step", "t"])


def main(statsfile, bs):
    data = parse_stats_file(statsfile)

    fig, ax = plt.subplots(figsize=(16, 10))
    for key, grp in data.groupby(["kind"]):
        ax = grp.plot(ax=ax, kind="line", x="step", y="t", label=key, marker="o")

    # adding title to the plot
    plt.title(f"Time per element by batcher, batch_size={bs}")

    # adding Label to the x-axis

    plt.xlabel("step")
    plt.ylabel("us")
    # adding legend to the curve
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    bs = sys.argv[2]

    main(filename, bs)
