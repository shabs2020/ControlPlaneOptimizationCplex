import networkx as nx
import json
import numpy as np
import random
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt


def draw_topology(graph, filename):
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="tab:blue",
        node_size=150,
        font_size=8,
        font_weight="bold",
    )
    plt.savefig(filename + ".png")
    plt.clf()


def create_network_from_excel(excel_file):
    df_nodes = pd.read_excel(excel_file, sheet_name="Nodes")
    df_links = pd.read_excel(excel_file, sheet_name="Links")
    graph = nx.Graph()
    for n in range(0, len(df_nodes)):
        graph.add_node(
            df_nodes["Name"][n],
            lon=df_nodes["Longitude"][n],
            lat=df_nodes["Latitude"][n],
            pos=(df_nodes["Longitude"][n], df_nodes["Latitude"][n]),
        )
    for e in range(0, len(df_links)):
        graph.add_edge(
            df_links["Node-A"][e],
            df_links["Node-Z"][e],
            linkDist=round(df_links["Length"][e], 3),
        )
    for e in graph.edges:
        link_cost = 1
        graph[e[0]][e[1]]["linkCost"] = link_cost

    for n in graph.nodes:
        graph.nodes[n]["demandVolume"] = (
            1.86 + 1.18 + (0.86 + 0.745 + 0.2 + 0.1) * graph.degree(n)
        )
    # pos= nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True,node_color='skyblue', node_size=220, font_size=8, font_weight="bold")
    # plt.show()
    return graph


def create_network(node_file, link_file):
    with open(node_file) as nF:
        nodes = json.load(nF)
    with open(link_file) as ef:
        edges = json.load(ef)
    graph = nx.Graph()
    for n in nodes:
        graph.add_node(
            nodes[n][0],
            lon=nodes[n][1],
            lat=nodes[n][2],
            pos=(nodes[n][1], nodes[n][2]),
            num_of_IXPs=nodes[n][3],
            num_of_DCs=nodes[n][4],
            ctraffic=1,
        )
    for e in edges:
        graph.add_edge(
            edges[e]["startNode"],
            edges[e]["endNode"],
            linkDist=round(edges[e]["linkDist"], 2),
            noChannels=edges[e]["noChannels"],
            noSpans=edges[e]["noSpans"],
            spanList=edges[e]["spanList"],
            capacity=2,
            used_capacity=0,
        )

    # Add unit cost of traffic
    for e in graph.edges:
        link_cost = 1
        graph[e[0]][e[1]]["linkCost"] = link_cost

    for n in graph.nodes:
        graph.nodes[n]["demandVolume"] = (
            1.86 + 1.18 + (0.86 + 0.745 + 0.2 + 0.1) * graph.degree(n)
        )
    return graph


# Random selection of M direct nodes and O indirect nodes
def select_nodes(network: nx.Graph, M: int):
    d_nodes = np.random.choice(network.nodes, M, replace=False).tolist()
    ind_nodes = list(set(list(network.nodes)).difference(d_nodes))
    return d_nodes, ind_nodes


# Select all possible combinations of direct nodes


def select_node_combinations(network: nx.Graph, M: int):
    network_nodes = list(network.nodes)
    combos = []
    for selection in combinations(network_nodes, M):
        combos.append(selection)

    return combos


# For each i, find a list of node disjoint paths to all direct nodes j
def find_potential_paths(i:str, edges: dict, network: nx.Graph):
    potential_paths = {}
    potential_path_edges = {}
    path_lengths = {}
    count = 1
    for j in network.nodes:
        if i!=j:
            for p in nx.all_simple_paths(G=network, source=i, target=j, cutoff=10):
                path_edges = []
                potential_paths["p" + str(count)] = p
                pairs = [p[i : i + 2] for i in range(len(p) - 1)]
                p_length = [network[i[0]][i[1]]["linkDist"] for i in pairs]
                path_lengths["p" + str(count)] = sum(p_length)
                for pair in pairs:
                    path_edges.append(
                        edges[(pair[0], pair[1])][0]
                        if (pair[0], pair[1]) in edges.keys()
                        else edges[(pair[1], pair[0])][0]
                    )
                potential_path_edges["p" + str(count)] = path_edges
                count += 1
            del path_edges
            del pairs

    return potential_paths, potential_path_edges, path_lengths


def define_control_demands(network: nx.Graph, edges: dict, scale_factor: int
):
    
    demand_volume = {}
    demand_paths = {}
    demand_path_edges = {}
    demand_path_lengths = {}
    for n in network.nodes:
        # demand_volume['d_'+n] = (1000/600)*((0.03+0.083) + (0.041+0.031) + (0.22+0.46 + 0.30+0.28))*network.degree(n)*scale_factor
        demand_volume["d_" + n] = network.nodes[n]["demandVolume"] * scale_factor
        potential_paths, potential_path_edges, path_lengths = find_potential_paths(n, edges, network)
        demand_paths["d_" + n] = potential_paths
        demand_path_edges["d_" + n] = potential_path_edges
        demand_path_lengths["d_" + n] = path_lengths
        del potential_paths, potential_path_edges, path_lengths

    return demand_volume, demand_paths, demand_path_edges, demand_path_lengths
# 


# pos= nx.spring_layout(graph)
# nx.draw(graph, pos, with_labels=True,node_color='skyblue', node_size=220, font_size=8, font_weight="bold")
# plt.savefig("TopologyVisual.png")
# plt.clf()


def round_capacity(capacity: float):
    rounded_value = 10.0
    if capacity > 10.0 and capacity <= 100.0:
        rounded_value = 100.0
    elif capacity > 100.0 and capacity <= 250.0:
        rounded_value = 250.0
    elif capacity > 250.0 and capacity <= 500.0:
        rounded_value = 500.0
    elif capacity > 500.0 and capacity <= 1000.0:
        rounded_value = 1000.0
    elif capacity > 1000.0 and capacity <= 2500.0:
        rounded_value = 1000.0
    elif capacity > 2500.0 and capacity <= 10000.0:
        rounded_value = 10000.0
    elif capacity > 10000.0 and capacity <= 25000.0:
        rounded_value = 25000.0
    elif capacity > 25000.0 and capacity <= 40000.0:
        rounded_value = 40000.0
    elif capacity > 40000.0 and capacity <= 100000.0:
        rounded_value = 100000.0
    elif capacity > 100000.00 and capacity <= 200000.0:
        rounded_value = 200000.0
    elif capacity > 200000.0 and capacity <= 400000.0:
        rounded_value = 400000.0
    return rounded_value
