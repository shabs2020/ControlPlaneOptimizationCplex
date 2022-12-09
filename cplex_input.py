import networkx as nx
import json
import numpy as np
import random
from itertools import combinations
import matplotlib.pyplot as plt


def create_network(node_file, link_file):
    with open(node_file) as nF:
        nodes = json.load(nF)
    with open(link_file) as ef:
        edges = json.load(ef)
    graph = nx.Graph()
    for n in nodes:
        graph.add_node(nodes[n][0], lon=nodes[n][1], lat=nodes[n][2], pos=(nodes[n][1], nodes[n][2]),
                       num_of_IXPs=nodes[n][3],
                       num_of_DCs=nodes[n][4], ctraffic=1)
    for e in edges:
        graph.add_edge(edges[e]['startNode'], edges[e]['endNode'], linkDist=round(edges[e]['linkDist'], 2),
                       noChannels=edges[e]['noChannels'], noSpans=edges[e]['noSpans'],
                       spanList=edges[e]['spanList'],
                       capacity=2, used_capacity=0)

    # Add unit cost of traffic
    for e in graph.edges:
        link_cost = (graph[e[0]][e[1]]['linkDist'] / 10) * \
            graph[e[0]][e[1]]['noSpans']
        graph[e[0]][e[1]]['linkCost'] = link_cost
    return graph


# Random selection of M direct nodes and O indirect nodes
def select_nodes(network: nx.Graph, M:int):
    d_nodes= np.random.choice(network.nodes,M,replace=False).tolist()
    ind_nodes=list(set(list(network.nodes)).difference(d_nodes))
    return d_nodes, ind_nodes

# Select all possible combinations of direct nodes
def select_node_combinations(network:nx.Graph, M:int):
    network_nodes=list(network.nodes)
    combos = list(combinations(network_nodes, M))
    return combos

    
# For each i, find a list of node disjoint paths to all direct nodes j 
def find_potential_paths(i: str, direct_nodes: list, edges: dict, network: nx.Graph):
    potential_paths={}
    potential_path_edges={}
    path_lengths={}
    count=1
    for j in direct_nodes:
        paths=list(nx.node_disjoint_paths(G=network,s=i,t=j,cutoff=5))
        for p in paths:
            path_edges = []
            potential_paths['p'+str(count)]=p
            pairs = [p[i: i + 2] for i in range(len(p)-1)]
            p_length=[network[i[0]][i[1]]['linkDist'] for i in pairs]
            path_lengths['p' + str(count)] = sum(p_length)
            for pair in pairs:
                path_edges.append(edges[(pair[0],pair[1])][0] if (pair[0],pair[1]) in edges.keys() else edges[(pair[1],pair[0])][0] )
            potential_path_edges['p'+str(count)] = path_edges
            count+=1
        del path_edges
        del pairs
        
    return potential_paths, potential_path_edges, path_lengths

def define_control_demands(nodes: list, direct_nodes: list, edges: dict, network: nx.Graph, scale_factor:int):
    demand_volume= {}
    demand_paths={}
    demand_path_edges={}
    demand_path_lengths={}
    for n in nodes:
        #demand_volume['d_'+n] = (1000/600)*((0.03+0.083) + (0.041+0.031) + (0.22+0.46 + 0.30+0.28))*network.degree(n)*scale_factor
        demand_volume['d_'+n] = (1.9+1.2 +0.86+0.745)*network.degree(n)*scale_factor
        potential_paths, potential_path_edges, path_lengths = find_potential_paths(n, direct_nodes, edges, network)

        demand_paths['d_'+n] = potential_paths
        demand_path_edges['d_'+n]=potential_path_edges
        demand_path_lengths['d_'+n] = path_lengths

    return demand_volume, demand_paths, demand_path_edges, demand_path_lengths
        



# pos= nx.spring_layout(graph)
# nx.draw(graph, pos, with_labels=True,node_color='skyblue', node_size=220, font_size=8, font_weight="bold")
# plt.savefig("TopologyVisual.png")
# plt.clf()
