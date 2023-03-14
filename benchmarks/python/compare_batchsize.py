import sys

import matplotlib.pyplot as plt
import pandas
import seaborn

KIND = "dynamic"


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "batchsize", "ms", "stddev"])


def main(fs, suffixes, iterations, outfile=None):
    dfs = [parse_stats_file(f) for f in fs]

    base_suffix = suffixes[0]
    base = dfs[0]
    base = dfs[0].rename(
        columns={"ms": f"ms_{base_suffix}", "stddev": f"stddev_{base_suffix}"}
    )
    for df, suffix in zip(dfs[1:], suffixes[1:]):
        df = df.rename(columns={"ms": f"ms_{suffix}", "stddev": f"stddev_{suffix}"})
        base = pandas.DataFrame.merge(
            base, df, on=["kind", "batchsize"], how="outer", suffixes=[""]
        )

    df = base
    print(df)
    base_suffix = suffixes[0]
    for suffix in suffixes[1:]:
        df[f"ratio_{suffix}"] = df[f"ms_{suffix}"] / df[f"ms_{base_suffix}"]

    df_melted = pandas.melt(
        df,
        id_vars=["kind", "batchsize"],
        value_vars=[f"ms_{suffix}" for suffix in suffixes],
    )

    nrows = 1 + len(dfs)
    fig, ax = plt.subplots(nrows=nrows, figsize=(16, 16 * nrows / 3))

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

    for suffix in suffixes:
        df[f"ms_{suffix}"] = df[f"ms_{suffix}"] * df["batchsize"]

    df_melted = pandas.melt(
        df,
        id_vars=["kind", "batchsize"],
        value_vars=[f"ms_{suffix}" for suffix in suffixes],
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

    for idx, suffix in enumerate(suffixes[1:]):
        df_melted = pandas.melt(
            df, id_vars=["kind", "batchsize"], value_vars=[f"ratio_{suffix}"]
        )
        seaborn.barplot(
            x="batchsize",
            y="value",
            hue="variable",
            data=df_melted.where(df_melted["kind"] == KIND),
            ax=ax[2 + idx],
        ).set(
            title=f"Speed ratio (<1 = faster), compared to {base_suffix}",
            ylabel="Relative speed",
            ylim=(0.5, 2.0),
        )
        seaborn.lineplot(x=[0, 24], y=[1, 1], color="red", ax=ax[2 + idx])

    if outfile:
        plt.savefig(outfile)

    else:
        plt.show()


def get_or_none(lst, idx):
    if idx >= len(lst):
        return None

    return lst[idx]


if __name__ == "__main__":
    filename = sys.argv[1].split(",")
    suffixes = sys.argv[2].split(",")
    bs = sys.argv[3]

    main(filename, suffixes, bs, get_or_none(sys.argv, 4))
