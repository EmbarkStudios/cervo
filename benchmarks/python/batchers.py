import sys

import matplotlib.pyplot as plt
import pandas
import seaborn


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "step", "t"])


def main(statsfile, bs):
    df = parse_stats_file(statsfile)
    seaborn.lineplot(x="step", y="t", hue="kind", data=df).set(
        title=f"Time per element by batcher, batch_size={bs}", ylabel="µs"
    )
    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    bs = sys.argv[2]

    main(filename, bs)
