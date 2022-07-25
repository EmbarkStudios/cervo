import sys

import matplotlib.pyplot as plt
import pandas
import seaborn


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "batchsize", "ms", "stddev"])


def main(statsfile1, statsfile2, iterations, outfile=None):
    df1 = parse_stats_file(statsfile1)
    first = df1.where(df1["kind"] == "fixed")
    first["kind"] = "before"

    df2 = parse_stats_file(statsfile2)
    second = df2.where(df2["kind"] == "fixed")
    second["kind"] = "after"

    df = pandas.concat([first, second], ignore_index=True)
    fig, ax = plt.subplots(nrows=2, figsize=(16, 16))

    seaborn.lineplot(x="batchsize", y="ms", hue="kind", data=df, ax=ax[0]).set(
        title=f"Mean execution time by batch size (concretized), its={iterations}",
        ylabel="milliseconds",
        ylim=(0, 1),
    )

    df["ms"] = df["ms"] * df["batchsize"]

    # seaborn.factorplot(x="batchsize", y="ms", col="kind", data=df, kind="bar").set(
    seaborn.lineplot(x="batchsize", y="ms", hue="kind", data=df, ax=ax[1]).set(
        title=f"Total execution time by batch size (concretized), its={iterations}",
        ylabel="milliseconds",
        ylim=(0, 10),
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
    filename2 = sys.argv[2]
    bs = sys.argv[3]

    main(filename, filename2, bs, get_or_none(sys.argv, 4))
