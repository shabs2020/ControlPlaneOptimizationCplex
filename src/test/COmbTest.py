import networkx as nx
import json
import numpy as np
import random
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import os


def create_network_from_excel(excel_file):
    df_nodes = pd.read_excel(excel_file, sheet_name='Nodes')
    df_links = pd.read_excel(excel_file, sheet_name='Links')
    graph = nx.Graph()
    for n in range(0, len(df_nodes)):
        graph.add_node(df_nodes['Name'][n], lon=df_nodes['Longitude'][n], lat=df_nodes['Latitude'][n], pos=(
            df_nodes['Longitude'][n], df_nodes['Latitude'][n]))
    for e in range(0, len(df_links)):
        graph.add_edge(df_links['Node-A'][e], df_links['Node-Z'][e], linkDist=round(df_links['Length'][e],3))
    # pos= nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True,node_color='skyblue', node_size=220, font_size=8, font_weight="bold")
    # plt.show()
    return graph

file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(file_path)

excel_file = BASE_DIR + '/Topologies/Coronet60.xlsx'
node_file = BASE_DIR + '/example_nodes.json'
edge_file = BASE_DIR + "/example_links.json"
#network = cplex_input.create_network(node_file, edge_file)

network=create_network_from_excel(excel_file)
links = {}
i = 0
for e in network.edges:
    links[e] = ['e'+str(i), 1]
    i += 1
    
network_nodes = list(network.nodes)

combos=[]
for selection in combinations(network_nodes, 30):
    print(list(selection))