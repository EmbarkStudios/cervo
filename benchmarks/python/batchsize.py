import sys

import matplotlib.pyplot as plt
import pandas
import seaborn


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "batchsize", "ms", "stddev"])


def main(statsfile, iterations, outfile=None):
    df = parse_stats_file(statsfile)
    fig, ax = plt.subplots(nrows=2, figsize=(16, 16))

    seaborn.lineplot(x="batchsize", y="ms", hue="kind", data=df, ax=ax[0]).set(
        title=f"Mean execution time by batch size (concretized), its={iterations}",
        ylabel="milliseconds",
        ylim=(0, 0.5),
    )

    df["ms"] = df["ms"] * df["batchsize"]

    #    seaborn.factorplot(x="batchsize", y="ms", col="kind", data=df, kind="bar").set(
    seaborn.lineplot(x="batchsize", y="ms", hue="kind", data=df, ax=ax[1]).set(
        title=f"Total execution time by batch size (concretized), its={iterations}",
        ylabel="milliseconds",
        ylim=(0, 3),
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
