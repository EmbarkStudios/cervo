import sys

import matplotlib.pyplot as plt
import pandas
import seaborn

KIND = "fixed"


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "batchsize", "ms", "stddev"])


def main(statsfile1, statsfile2, iterations, outfile=None):
    df1 = parse_stats_file(statsfile1)
    df2 = parse_stats_file(statsfile2)

    df = pandas.DataFrame.merge(
        df1, df2, on=["kind", "batchsize"], how="outer", suffixes=["_before", "_after"]
    )
    df["ratio"] = df["ms_after"] / df["ms_before"]

    df_melted = pandas.melt(
        df, id_vars=["kind", "batchsize"], value_vars=["ms_before", "ms_after"]
    )

    fig, ax = plt.subplots(nrows=3, figsize=(16, 16))

    seaborn.lineplot(
        x="batchsize",
        y="value",
        hue="variable",
        data=df_melted.where(df_melted["kind"] == KIND),
        ax=ax[0],
    ).set(
        title=f"Mean execution time by batch size (concretized), its={iterations}",
        ylabel="milliseconds",
        ylim=(0, 1),
    )

    df["ms_before"] = df["ms_before"] * df["batchsize"]
    df["ms_after"] = df["ms_after"] * df["batchsize"]

    df_melted = pandas.melt(
        df, id_vars=["kind", "batchsize"], value_vars=["ms_before", "ms_after"]
    )

    seaborn.lineplot(
        x="batchsize",
        y="value",
        hue="variable",
        data=df_melted.where(df_melted["kind"] == KIND),
        ax=ax[1],
    ).set(
        title=f"Total execution time by batch size (concretized), its={iterations}",
        ylabel="milliseconds",
        ylim=(0, 10),
    )

    df_melted = pandas.melt(df, id_vars=["kind", "batchsize"], value_vars=["ratio"])
    seaborn.barplot(
        x="batchsize",
        y="value",
        hue="variable",
        data=df_melted.where(df_melted["kind"] == KIND),
        ax=ax[2],
    ).set(
        title=f"Relative improvement",
        ylabel="Relative speed",
        ylim=(0.6, 1.4),
    )

    # print(df.where(df["partition"] == "before")["ms"])
    # r = (
    #     df.where(df["partition"] == "before")["ms"]
    #     / df.where(df["partition"] == "after")["ms"]
    # )
    # print(r)
    # seaborn.lineplot(x=list(range(1, 25)), y=r, data=df, ax=ax[2],).set(
    #     title=f"Total execution time by batch size (concretized), its={iterations}",
    #     ylabel="milliseconds",
    #     ylim=(0, 10),
    # )

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
