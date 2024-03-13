import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from model import BoundedConfidence


def visualize(model: BoundedConfidence, steps: int, digits: int = 2):
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
    colormap = plt.get_cmap("cool")  # other options: rainbow

    # add nodes
    for step in range(steps):
        opinions = results.loc[step, :].drop_duplicates()
        for opinion in opinions:
            net.add_node(
                (opinion, step),
                pos=(step, opinion),
                color=mpl.colors.to_hex(colormap(norm(opinion))),
            )
    pos = nx.get_node_attributes(net, "pos")
    colors = list(nx.get_node_attributes(net, "color").values())
    labels = {node: round(node[0], digits) for node in net.nodes()}

    # add edges
    for step in range(steps - 1):
        for agent in model.agents:
            source_node = (results.at[step, agent], step)
            target_node = (results.at[step + 1, agent], step + 1)
            edge = (source_node, target_node)
            if edge not in net.edges():
                net.add_edge(*edge)

    # draw
    options = {"node_size": 1000, "font_family": "SegUI", "font_size": 12}
    nx.draw(net, pos=pos, node_color=colors, **options)
    nx.draw_networkx_labels(net, pos, labels)


start = [0.1, 0.2, 0.3, 0.35]
model = BoundedConfidence(start, confidence_threshold=0.1)
visualize(model, 5)
