import sys

import matplotlib.pyplot as plt
import pandas
import seaborn


def parse_stats_file(statsfile) -> pandas.DataFrame:
    return pandas.read_csv(statsfile, names=["kind", "step", "t"])


def main(statsfile, bs):
    df = parse_stats_file(statsfile)
    seaborn.lineplot(x="step", y="t", hue="kind", data=df).set(
        title=f"Time per element by noise quality, batch_size={bs}", ylabel="µs"
    )
    plt.show()


#    # adding title to the plot
#    plt.title(")
#
#    # adding Label to the x-axis
#
#    plt.xlabel("step")
#    plt.ylabel("us")
#    # adding legend to the curve
#
#    plt.show()
#

if __name__ == "__main__":
    filename = sys.argv[1]
    bs = sys.argv[2]

    main(filename, bs)
