import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["format", "kind", "t", "stddev"])


def main(statsfile, bs):
    df = parse_stats_file(statsfile)

    seaborn.barplot(x="kind", y="t", hue="format", data=df).set(
        title=f"Mean load time by format and kind, batch_size={bs}",
        ylabel="milliseconds",
    )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=3,
    )

    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    bs = sys.argv[2]

    main(filename, bs)
