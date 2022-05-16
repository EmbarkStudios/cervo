import sys

import matplotlib.pyplot as plt
import pandas
import seaborn


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "step", "t"])


def main(statsfile, bs, outfile=None):
    df = parse_stats_file(statsfile)
    plt.figure(figsize=(16, 10))
    seaborn.lineplot(x="step", y="t", hue="kind", data=df).set(
        title=f"Time per element by batcher, batch_size={bs}", ylabel="µs"
    )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=3,
    )

    if outfile:
        plt.savefig(outfile)

    else:
        plt.show()


def get_or_none(lst, idx):
    if idx >= len(lst):
        return None

    return lst[idx]


if __name__ == "__main__":
    filename = sys.argv[1]
    bs = sys.argv[2]

    main(filename, bs, get_or_none(sys.argv, 3))
