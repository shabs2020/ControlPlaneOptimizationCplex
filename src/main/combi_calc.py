import networkx as nx
import json
import numpy as np
import random
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import cplex_input
import os
file_path = os.path.abspath(os.path.join(__file__, "../.."))
BASE_DIR = os.path.dirname(file_path) + '/'
network=cplex_input.create_network(node_file= BASE_DIR + 'Topologies/Nodes_Germany_17.json', link_file=BASE_DIR+'Topologies/Links_Germany_17.json')
total_episodes = len(network.nodes)
count=0
network_nodes = list(network.nodes)
for m in range(2, total_episodes + 1):
    for d_node_combos in combinations(network_nodes, m):
        count+=1
print(count)
