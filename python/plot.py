import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

DATA_OLD = [
    [1,      1,      303],
    [2,      1,      296],
    [1,      2,      273],
    [2,      2,      155],
    [3,      2,      279],
    [1,      3,      263],
    [2,      3,      234],
    [3,      3,      99],
    [4,      3,      269],
    [1,      4,      263],
    [2,      4,      147],
    [3,      4,      173],
    [4,      4,      83],
    [5,      4,      258],
    [1,      5,      261],
    [2,      5,      186],
    [3,      5,      188],
    [4,      5,      137],
    [5,      5,      62],
    [6,      5,      256],
    [1,      6,      254],
    [2,      6,      145],
    [3,      6,      100],
    [4,      6,      153],
    [5,      6,      117],
    [6,      6,      52],
    [7,      6,      255],
    [1,      7,      255],
    [2,      7,      171],
    [3,      7,      139],
    [4,      7,      175],
    [5,      7,      137],
    [6,      7,      97],
    [7,      7,      77],
    [8,      7,      264],
    [1,      8,      261],
    [2,      8,      148],
    [3,      8,      168],
    [4,      8,      79],
    [5,      8,      161],
    [6,      8,      128],
    [7,      8,      112],
    [8,      8,      71],
    [9,      8,      265],
    [1,      9,      258],
    [2,      9,      178],
    [3,      9,      104],
    [4,      9,      116],
    [5,      9,      170],
    [6,      9,      146],
    [7,      9,      137],
    [8,      9,      103],
    [9,      9,      60],
    [10,     9,      259],
    [1,      10,     260],
    [2,      10,     148],
    [3,      10,     135],
    [4,      10,     134],
    [5,      10,     65],
    [6,      10,     153],
    [7,      10,     147],
    [8,      10,     121],
    [9,      10,     96],
    [10,     10,     55],
    [11,     10,     260],
]

DATA = [
    [1,  1,  720],
    [2,  1,  693],
    [1,  2,  631],
    [2,  2,  384],
    [3,  2,  645],
    [1,  3,  630],
    [2,  3,  645],
    [3,  3,  649],
    [4,  3,  645],
    [1,  4,  668],
    [2,  4,  474],
    [3,  4,  537],
    [4,  4,  291],
    [5,  4,  613],
    [1,  5,  637],
    [2,  5,  616],
    [3,  5,  775],
    [4,  5,  457],
    [5,  5,  241],
    [6,  5,  597],
    [1,  6,  578],
    [2,  6,  381],
    [3,  6,  496],
    [4,  6,  465],
    [5,  6,  364],
    [6,  6,  214],
    [7,  6,  567],
    [1,  7,  699],
    [2,  7,  476],
    [3,  7,  535],
    [4,  7,  463],
    [5,  7,  405],
    [6,  7,  333],
    [7,  7,  695],
    [8,  7,  592],
    [1,  8,  583],
    [2,  8,  379],
    [3,  8,  489],
    [4,  8,  239],
    [5,  8,  422],
    [6,  8,  357],
    [7,  8,  358],
    [8,  8,  578],
    [9,  8,  578],
    [1,  9,  562],
    [2,  9,  461],
    [3,  9,  383],
    [4,  9,  340],
    [5,  9,  439],
    [6,  9,  381],
    [7,  9,  710],
    [8,  9,  308],
    [9,  9,  203],
    [10, 9,  557],
    [1,  10, 562],
    [2,  10, 383],
    [3,  10, 491],
    [4,  10, 454],
    [5,  10, 210],
    [6,  10, 417],
    [7,  10, 700],
    [8,  10, 383],
    [9,  10, 304],
    [10, 10, 222],
    [11, 10, 592],
]

agent_counts = {idx: [] for idx in range(1, 11)}

# for record in DATA:
#     agent_counts[record[1]].append((record[0], record[2]))

# print(agent_counts[1])
# print(agent_counts[2])

# print()

# labels = []
# axes = []
# for agent_count, metrics in agent_counts.items():
#     xs, ys = list(zip(*metrics))
#     axes.append(plt.plot(xs, list(y / 1e3 for y in ys)))
#     labels.append(f"{agent_count} agents")

batch_size_data = {v: [0 for _ in range(11)] for v in range(1, 12)}
for record in DATA:
    batch_size_data[record[0]][record[1] - 1] = record[2] / 1000

width = 0.05

fig = plt.figure()
ax = fig.add_subplot(111)
indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
group_idx = 0
for group, data in batch_size_data.items():
    x_pos = indices + width * group_idx
    print(x_pos, data)
    ax.bar(x_pos, data, width)
    group_idx += 1


plt.xlabel("agent count")
plt.ylabel("micros")
plt.title("time per element (cmp by batchsize)")
plt.savefig("agentcount.png")
plt.xticks(ticks=[v for v in range(1, 11)], labels=[str(v) for v in range(1, 11)])
plt.clf()

batch_size_data = {v: [0 for _ in range(11)] for v in range(1, 12)}
for record in DATA:
    batch_size_data[record[1]][record[0] - 1] = record[2] / 1000

width = 0.05

fig = plt.figure()
ax = fig.add_subplot(111)
indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
group_idx = 0
for group, data in batch_size_data.items():
    x_pos = indices + width * group_idx
    print(x_pos, data)
    ax.bar(x_pos, data, width)
    group_idx += 1


plt.xlabel("batch size")
plt.ylabel("micros")
plt.title("time per element (cmp by agentcount)")
plt.savefig("batchsize.png")
plt.xticks(ticks=[v for v in range(1, 11)], labels=[str(v) for v in range(1, 11)])
plt.clf()
cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

        'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }

# Create the colormap using the dictionary
GnRd = colors.LinearSegmentedColormap('GnRd', cdict)

fig, ax = plt.subplots()
data = np.array(DATA)
ax.scatter(data[:, 0 ], data[:, 1], cmap=GnRd, c=data[:, 2], s=data[:, 2])

ax.set_xlabel("agent count")
ax.set_ylabel("batch size")
ax.set_title(f"time per element, max={max(data[:, 2])} us, min={min(data[:, 2])} us")
ax.grid(True)
fig.tight_layout()
plt.savefig("scatter.png")
plt.clf()

fig, ax = plt.subplots()
data = np.array(DATA_OLD)
ax.scatter(data[:, 0 ], data[:, 1], cmap=GnRd, c=data[:, 2], s=data[:, 2])

ax.set_xlabel("agent count")
ax.set_ylabel("batch size")
ax.set_title(f"time per element, max={max(data[:, 2])} us, min={min(data[:, 2])} us")
ax.grid(True)
fig.tight_layout()
plt.savefig("scatter_old.png")
plt.clf()


fig, ax = plt.subplots()
data = np.array(DATA)
dataold = np.array(DATA_OLD)
delta = data[:,2] / dataold[:,2] * 50.0
ax.scatter(data[:, 0 ], data[:, 1], cmap=GnRd, c=delta, s=delta)

mx = max(delta) * 2
mn = min(delta) * 2
ax.set_xlabel("agent count")
ax.set_ylabel("batch size")
ax.set_title(f"relative cost, max={int(mx)} %, min={int(mn)} %")
ax.grid(True)
fig.tight_layout()
plt.savefig("diff.png")
plt.clf()

