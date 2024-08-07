import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

from model import BoundedConfidence


def visualize(
    model: BoundedConfidence,
    steps: int,
    digits: int = 3,
    blackwhite=False,
    output_file=None,
):
    # get model run results
    results = model.run(number_of_steps=steps)
    results.drop_duplicates(inplace=True)
    steps = len(results.index)

    # initialize
    net = nx.DiGraph()

    # add nodes
    for step in range(steps):
        opinions = results.loc[step, :].drop_duplicates()
        for opinion in opinions:
            net.add_node((opinion, step))
    pos = {node: (node[1], node[0]) for node in net.nodes()}

    # add edges
    for step in range(steps - 1):
        for agent in model.agents:
            source_node = (results.at[step, agent], step)
            target_node = (results.at[step + 1, agent], step + 1)
            edge = (source_node, target_node)
            if edge not in net.edges():
                net.add_edge(*edge)

    # colors
    cmap = "coolwarm"  # other options: coolwarm, Greys
    if blackwhite:
        cmap = "Greys"
    colormap = ListedColormap(cm.get_cmap(cmap)(np.linspace(0.3, 0.7)))
    color_nodes = [node[0] for node in net.nodes()]
    labels = {node: round(node[0], digits) for node in net.nodes()}
    if digits == 0:
        labels = {node: int(node[0]) for node in net.nodes()}

    # draw
    options = {"node_size": 1000, "font_family": "SegUI", "font_size": 12}
    plt.clf()
    nx.draw(
        net,
        pos=pos,
        node_color=color_nodes,
        # edge_color=color_edges,
        cmap=colormap,
        **options
    )

    nx.draw_networkx_labels(net, pos, labels)

    if output_file:
        plt.savefig(fname=output_file, dpi="figure")
    else:
        plt.show()


if __name__ == "__main__":
    start = [0.1, 0.2, 0.3, 0.35]
    model = BoundedConfidence(start, confidence_threshold=0.1)
    visualize(model, 5)
