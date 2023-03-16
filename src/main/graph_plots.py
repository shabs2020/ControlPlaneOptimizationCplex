import networkx as nx
import matplotlib.pyplot as plt
import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os
import pandas as pd 
file_path = os.path.abspath(os.path.join(__file__, "../.."))
BASE_DIR = os.path.dirname(file_path)

demands_sheet=pd.read_excel(BASE_DIR + '/Stats/Model_Stats_LN1_M2_1.xlsx', sheet_name='Demands',header=0)
nodes=[]
direct_nodes=[i for i in demands_sheet['Destination'].unique() ]
nodes.extend(direct_nodes)
nodes.extend(i for  i in demands_sheet['Source'].unique())
nodes=[i for i in nodes if pd.notnull(i)]
links_sheet=pd.read_excel(BASE_DIR + '/Stats/Model_Stats_LN1_M2_1.xlsx', sheet_name='Links',header=0)

network=nx.Graph()
for n in nodes:
    network.add_node(n, id=nodes.index(n))

for e in range(len(links_sheet['No.'])):
    network.add_edge(links_sheet['FirstEnd'][e], links_sheet['SecondEnd'][e], id=links_sheet['Links_Ids'][e])

color_map = ['red' if node in direct_nodes else 'green' for node in network.nodes]        
edge_labels = dict([((n1, n2), d['id'])
                    for n1, n2, d in network.edges(data=True)])

pos= nx.spring_layout(network)
nx.draw(network, pos, with_labels=True,node_color=color_map, node_size=50, font_size=5, font_weight="bold")
nx.draw_networkx_edge_labels(network, pos,edge_labels=edge_labels,font_size=5,font_color='blue')
plt.show()

path_index_per_demand= demands_sheet[~demands_sheet['Source'].isnull()].index.tolist()
routes=[]
for elem,next_elem in zip(path_index_per_demand, path_index_per_demand[1:]+[path_index_per_demand[0]]):
    if demands_sheet['Source'][elem]=='Tampa':
        destination=list(set([demands_sheet['Destination'][i] for i in range(elem, next_elem)]))
        demand=[i for i in range(elem, next_elem) if demands_sheet['Source'][i]]
        for i in range(elem, next_elem):
            routes.append(demands_sheet['Paths'][i].split("-"))

path_index_per_demand= demands_sheet[~demands_sheet['Source'].isnull()].index.tolist()
routes=[]


for elem,next_elem in zip(path_index_per_demand, path_index_per_demand[1:]+[path_index_per_demand[0]]):
    if demands_sheet['Source'][elem]=='Tampa':
        destination=list(set([demands_sheet['Destination'][i] for i in range(elem, next_elem)]))
        demand=[i for i in range(elem, next_elem) if demands_sheet['Source'][i]]
        for i in range(elem, next_elem):
            routes.append(demands_sheet['Paths'][i].split("-"))
            #paths[(demands_sheet['Path Ids'][i])]=demands_sheet['Paths'][i].split("-") 

G=nx.Graph(name="buba")
edges = []
color_count=0
for r in routes:
    #route_edges = [(paths[r][n],paths[r][n+1]) for n in range(len(paths[r])-1)]
    route_edges = [(r[n],r[n+1]) for n in range(len(r)-1)]
    G.add_nodes_from(r)
    G.add_edges_from(route_edges)
    # if destination.index(paths[r][-1]) >= len(colors):
    #     color_index=destination.index(paths[r][-1])-len(colors)
    # else:
    #     color_index=destination.index(paths[r][-1])
    # for e in route_edges:
    #     G.add_edge(e[0],e[1])
    

edges=G.edges()
# path_per_dest={}
# for d in destination:
#     all_paths=[]
#     for edge_l in edges:
#         if edge_l[-1][1] == d:
#             all_paths.append(edge_l)

#     path_per_dest[d]=all_paths



print("Graph has %d nodes with %d edges" %(G.number_of_nodes(),    
G.number_of_edges()))
nodes=[node for node in G.nodes]
nodes_dict = dict(zip(nodes, nodes))
pos= nx.spring_layout(G)
plt.figure(figsize=(8,4))
#layout = nx.spring_layout(G,k=0.1,iterations=len(G.nodes))
layout=pos

edge_labels_sub={}
for e in G.edges:
    if e in network.edges:
        edge_labels_sub[e]= network[e[0]][e[1]]['id']
    elif (e[1],e[0]) in network.edges:
        edge_labels_sub[e] = network[e[1]][e[0]]['id']
nx.draw(G, pos, with_labels=False,node_color='green', node_size=50, font_size=10, font_weight="bold")

nx.draw_networkx_edges(G,pos=pos,edgelist=G.edges)
nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels_sub,font_size=10,font_color='blue')

nx.draw_networkx(G.subgraph('Tampa'), pos=pos, with_labels=False, node_color='blue',node_size=80)
nx.draw_networkx(G.subgraph(i for i in destination), pos=pos, with_labels=False, font_size=10, node_color='red', font_color='black',node_size=80)
for l in layout:  # raise text positions
    layout[l][1] += 0.07 
    layout[l][0] -= 0.009 # probably small value enough
nx.draw_networkx_labels(G, layout, labels=nodes_dict)
plt.show()







