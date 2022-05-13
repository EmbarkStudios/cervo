import sys

import matplotlib.pyplot as plt
import pandas
import seaborn


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["format", "kind", "t", "stddev"])


def main(statsfile, its, outfile=None):
    df = parse_stats_file(statsfile)

    seaborn.barplot(x="kind", y="t", hue="format", data=df).set(
        title=f"Mean load time by format and kind, its={its}",
        ylabel="milliseconds",
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
    its = sys.argv[2]
    main(filename, its, get_or_none(sys.argv, 3))
