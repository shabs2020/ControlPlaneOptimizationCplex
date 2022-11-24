import networkx as nx
import json
import numpy as np
import random

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


# Random selection of M direct nodes that are also non-adjacent
def select_direct_nodes(network: nx.Graph, M:int):
    
    direct_nodes=np.random.choice(network.nodes,M,replace=False).tolist()
    # direct_nodes = []
    
    # while len(direct_nodes) != M:
    #     potential_node = random.choice(list(network.nodes))
    #     neighbours = [n for n in network.neighbors(potential_node)]
    #     if (not any(item in neighbours for item in direct_nodes)) & (potential_node not in direct_nodes):
    #         direct_nodes.append(potential_node)
    return direct_nodes


# For each i, sort all direct nodes j based on the shortest path distance
def find_shortest_paths(i: str, direct_nodes: list, edges: dict, network: nx.Graph):
    shortest_paths = {}
    for j in direct_nodes:

        shortest_paths[(i, j)] = [nx.shortest_path_length(
            network, source=i, target=j, weight='linkDist')]
        paths = nx.shortest_path(
            network, source=i, target=j, weight='linkDist')
        shortest_paths[(i, j)].append(paths)

    # Check if shortest path uses an edge that is common to all paths connecting node i to all other direct nodes.
    # In such a case, while checking the link disjoint paths, only one path will be returned
    # However the path diversity constraint will be violated with no. of paths < = 1.
    # Hence in the below section, we check is there is a common node in all paths.
    # If yes, then we try to select the second shortest path for the shortest path
    #print("direct_nodes {}".format(direct_nodes))
    #print("indirect node is {}".format(i))
    all_paths = [i[1][1:] for i in shortest_paths.values()]
   
    print(all_paths)
    common_node_in_all = list(set.intersection(*map(set, all_paths)))
    print(common_node_in_all)
    if len(common_node_in_all) > 0:
        first_key = [k for k in shortest_paths if min(
            i[0] for i in shortest_paths.values()) in shortest_paths[k]][0]
        #print("first_key {}".format(first_key))
        alt_paths = list(nx.node_disjoint_paths(
            network, s=first_key[0], t=first_key[1]))
        #print("alt_paths {}".format(alt_paths))
        selected_path = [
            sublist for sublist in alt_paths if sublist != shortest_paths[first_key][1]]
        if len(selected_path) >0:
            path_length = [nx.path_weight(
            network, i, weight="linkDist") for i in selected_path]
            #print("Selected path is {} and length is {} ".format(selected_path,path_length))
            selected_path = selected_path[path_length.index(min(path_length))]
            shortest_paths[first_key] = [min(path_length), selected_path]

    for p in shortest_paths:
        path_edges = []
        for edge in list(zip(shortest_paths[p][1], shortest_paths[p][1][1:])):
            path_edges.append(edges[edge][0]) if edge in edges.keys(
            ) else path_edges.append(edges[tuple(reversed(edge))][0])
        shortest_paths[p][1] = path_edges

    sorted_paths = sorted(shortest_paths.items(), key=lambda x: x[1][0])
    shortest_paths = {k: v for k, v in sorted_paths}

    return shortest_paths

def find_tuple_byValue(element_list:list, item_to_compare:list):
    item_found = [ x[1] for x, y in element_list if y[1]  == item_to_compare ]
    if len(item_found)>0:
        return item_found[0]
    else:
        return None

def get_disjoint_paths(shortest_paths: list, j: int):
    #print("shortest_paths {}".format(shortest_paths))
    s_paths = [i[1][1] for i in shortest_paths]
    print(s_paths)
    disjoint_paths = {}
    node_unit_cost=0
    while (len(s_paths)) > 0:
        if len(s_paths) == 1:
            disjoint_paths[('p'+str(j),find_tuple_byValue(shortest_paths,s_paths[0]))] = s_paths[0]
            node_unit_cost = len(s_paths[0]) + 1+node_unit_cost
            s_paths.pop(0)
        else:
            for p in s_paths[1:]:
                if any(check in s_paths[0] for check in p):
                    s_paths.remove(p)
            disjoint_paths[('p'+str(j),find_tuple_byValue(shortest_paths,s_paths[0]))] = s_paths[0]
            node_unit_cost = len(s_paths[0]) + 1 +node_unit_cost
            s_paths.pop(0)
        j += 1

    return disjoint_paths, j, node_unit_cost


def define_control_demands(nodes: list, direct_nodes: list, edges: dict, network: nx.Graph):
    # node cost and path names can be further included in this dictionary
    # set_demands[nodei] = {di, h(d), P(di), Cost(nodei)}
    set_demands = {}
    i = 0
    j = 0
    for n in nodes:
        set_demands[n] = ['d'+str(i), 4]
        shortest_paths = list(find_shortest_paths(
            str(n), direct_nodes, edges, network).items())
        #print("shortest_paths {}".format(shortest_paths))
        disjoint_paths, j,node_unit_cost = get_disjoint_paths(shortest_paths, j=j)
        print("disjoint_paths {}".format(disjoint_paths))
        set_demands[n].append(disjoint_paths)
        set_demands[n].append(node_unit_cost*2)
        i += 1
    del(shortest_paths)
    return set_demands


def create_input(node_file, edge_file):
    network = create_network(node_file, edge_file)

    direct_nodes = select_direct_nodes(network=network)
    indirect_nodes = set(list(network.nodes)).difference(direct_nodes)

    set_links = {}
    i = 1
    for e in network.edges:
        set_links[e] = ['e'+str(i), 1]
        i += 1

    set_demands = define_control_demands(
        indirect_nodes, direct_nodes, set_links, network)

    return direct_nodes, indirect_nodes, set_links, set_demands


# pos= nx.spring_layout(graph)
# nx.draw(graph, pos, with_labels=True,node_color='skyblue', node_size=220, font_size=8, font_weight="bold")
# plt.savefig("TopologyVisual.png")
# plt.clf()
