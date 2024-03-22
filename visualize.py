import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from model import BoundedConfidence


def visualize(model: BoundedConfidence, steps: int, digits: int = 3, gray=False):
    # get model run results
    results = model.run(number_of_steps=steps)
    results.drop_duplicates(inplace=True)
    steps = len(results.index)

    # initialize
    net = nx.DiGraph()
    norm = mpl.colors.Normalize(
        vmin=min(model.start_distribution),
        vmax=max(model.start_distribution),
    )

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
    if gray:
        color_nodes = ["gray" for node in net.nodes()]
        color_edges = ["gray" for edge in net.edges()]
        labels = {node: "" for node in net.nodes()}
    else:
        colormap = plt.get_cmap("cool")  # other options: rainbow
        color_nodes = [
            mpl.colors.to_hex(colormap(norm(node[0]))) for node in net.nodes()
        ]
        color_edges = ["k" for edge in net.edges()]
        labels = {node: round(node[0], digits) for node in net.nodes()}

    # draw
    options = {"node_size": 1000, "font_family": "SegUI", "font_size": 12}
    nx.draw(net, pos=pos, node_color=color_nodes, edge_color=color_edges, **options)
    nx.draw_networkx_labels(net, pos, labels)
