import networkx as nx
import matplotlib.pyplot as plt
import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os
import pandas as pd 
import numpy as np
file_path = os.path.abspath(os.path.join(__file__, "../.."))
BASE_DIR = os.path.dirname(file_path)

def network_plot(networkfname, nodename):

    demands_sheet=pd.read_excel(BASE_DIR + '/'+ networkfname, sheet_name='Demands',header=0)
    nodes=[]
    direct_nodes=[i for i in demands_sheet['Destination'].unique() ]
    nodes.extend(direct_nodes)
    nodes.extend(i for  i in demands_sheet['Source'].unique())
    nodes=[i for i in nodes if pd.notnull(i)]
    links_sheet=pd.read_excel(BASE_DIR + '/'+ networkfname, sheet_name='Links',header=0)

    network=nx.Graph()
    for n in nodes:
        network.add_node(n, id=nodes.index(n))

    for e in range(len(links_sheet['No.'])):
        network.add_edge(links_sheet['FirstEnd'][e], links_sheet['SecondEnd'][e], id=links_sheet['Links_Ids'][e])
    print(direct_nodes)
    
    color_map = ['brown' if node in direct_nodes else 'green' for node in network.nodes]        
    edge_labels = dict([((n1, n2), d['id'])
                        for n1, n2, d in network.edges(data=True)])
    plt.figure(figsize=(9, 5.7), dpi=80)
    pos= nx.spring_layout(network)
    nx.draw(network, pos, with_labels=True,node_color=color_map, node_size=100, font_size=11,font_weight="bold")
    nx.draw_networkx_edge_labels(network, pos,edge_labels=edge_labels,font_size=11,font_color='blue', font_weight="bold")
    #plt.tight_layout()
    plt.savefig(BASE_DIR + "/Figures/Germany17_new.png", format="png", pad_inches=0)
    del color_map

    path_index_per_demand= demands_sheet[~demands_sheet['Source'].isnull()].index.tolist()
    path_index_per_demand.append(len(demands_sheet['Source'])-1)
    print(path_index_per_demand)
    routes=[]
    for elem,next_elem in zip(path_index_per_demand, path_index_per_demand[1:]+[path_index_per_demand[0]]):
        if demands_sheet['Source'][elem]==nodename:
            destination=list(set([demands_sheet['Destination'][i] for i in range(elem, next_elem)]))
            demand=[i for i in range(elem, next_elem) if demands_sheet['Source'][i]]
            for i in range(elem, next_elem):
                routes.append(demands_sheet['Paths'][i].split("-"))
                #paths[(demands_sheet['Path Ids'][i])]=demands_sheet['Paths'][i].split("-") 
    print(routes)
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
    nx.draw(G, pos, with_labels=False,node_color='green', node_size=100, font_size=11,font_weight="bold")

    nx.draw_networkx_edges(G,pos=pos,edgelist=G.edges)
    nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels_sub,font_size=11,font_color='blue', font_weight="bold")

    nx.draw_networkx(G.subgraph(nodename), pos=pos, with_labels=False, node_color='blue',node_size=100)
    nx.draw_networkx(G.subgraph(i for i in destination), pos=pos, with_labels=False, font_size=11, node_color='brown', font_color='black',node_size=100, font_weight="bold")
    for l in layout:  # raise text positions
        layout[l][1] += 0.07 
        layout[l][0] -= 0.009 # probably small value enough
    nx.draw_networkx_labels(G, layout, labels=nodes_dict)
    plt.savefig(BASE_DIR + "/Figures/Germany17_new_subgraph_i.png", format="png", pad_inches=0)
    # plt.show()


def plot_total_capacity_costs(fname):
    solver_data=pd.read_excel(BASE_DIR + fname, sheet_name='Obj_Values',header=0)
    x_values=list(solver_data['Nodes Num.'])
    link_utils=solver_data['LinkBW']
    node_utils=solver_data['NodeCosts']
    node_utils=node_utils.fillna(solver_data['Obj. Value'].iloc[-1])
    net_capacity_costs=link_utils+node_utils
    y_values=list(net_capacity_costs)
    # print('link_utils{}'.format(link_utils))
    # print('node_utils {}'.format(node_utils))
    # print('net_capacity_costs {}'.format(net_capacity_costs))
    plt.figure(figsize=(8.5, 5), dpi=80)
    color = "tab:blue"
    plt.plot(x_values, y_values, '-o', color=color, linewidth=2)
    min_x = x_values[np.argmin(y_values)]
    min_y = min(y_values)
    plt.plot(min_x, min_y, "s", c="r" ,label="Minimum Capacity Costs min($C_{CP}$)")
   # plt.scatter(min_x, min_y, c="r" ,label="Minimum Capacity Costs")
    plt.legend(fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(1, 18, 1),fontsize=12)
    plt.xlabel("Number of control plane interfaces ($m$)", fontsize=18)
    plt.ylabel("Network  Capacity Costs [Mbps]", fontsize=18)
    plt.grid(axis = 'y')
    plt.tight_layout()
    plt.savefig(BASE_DIR + "/Figures/NetworkCostsCoronet30.png", format="png", pad_inches=0)

def plot_costs_contrib(fname):
    solver_data=pd.read_excel(BASE_DIR+ fname, sheet_name='Obj_Values',header=0)
    x_values=list(solver_data['Nodes Num.'])
    link_utils=list(solver_data['LinkBW'])
    node_utils=solver_data['NodeCosts']
    node_utils=list(node_utils.fillna(solver_data['Obj. Value'].iloc[-1]))
    plt.figure(figsize=(10, 6), dpi=100)
    color = "tab:blue"

    plt.plot(x_values, link_utils, color='g', linewidth=2, label='Capacity Costs due to link utilization ($C_{L}$)')
    plt.plot(x_values, node_utils, color='r', linewidth=2, label='Capacity Costs of control plane interfaces ($C_{C}$)')
    x1_annotations = [x_values[7]]
    y1_annotations = [link_utils[7]]
    y2_annotations = [node_utils[7]]
    # min_x = x_values[np.argmin(y_values)]
    # min_y = min(y_values)
    plt.plot(x1_annotations, y1_annotations, "o", c="g" )
    plt.plot(x1_annotations, y2_annotations, "o", c="r" )
    label1="{:.2f}".format(y1_annotations[0])
    plt.annotate(label1,  # this is the text
                      xy=(x1_annotations[0], y1_annotations[0]), xytext=(20,15), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.2),
            arrowprops=dict(arrowstyle='->', 
                            color='green'),fontsize=14)
    # label2="{:.2f}".format(y1_annotations[1])
    # plt.annotate(label2,  # this is the text
    #                   xy=(x1_annotations[1], y1_annotations[1]), xytext=(50,-10), 
    #         textcoords='offset points', ha='center', va='bottom',
    #         bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.2),
    #         arrowprops=dict(arrowstyle='->', 
    #                         color='green'),fontsize=14)
    label1="{:.2f}".format(y2_annotations[0])
    plt.annotate(label1,  # this is the text
                      xy=(x1_annotations[0], y2_annotations[0]), xytext=(10,-40), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.2),
            arrowprops=dict(arrowstyle='->',
                            color='red'),fontsize=14)
    # label2="{:.2f}".format(y2_annotations[1])
    # plt.annotate(label2,  # this is the text
    #                   xy=(x1_annotations[1], y2_annotations[1]), xytext=(20,30), 
    #         textcoords='offset points', ha='center', va='bottom',
    #         bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.2),
    #         arrowprops=dict(arrowstyle='->', 
    #                         color='red'),fontsize=14)
    # for x, y in zip(x1_annotations, y1_annotations):
    #     label = "{:.2f}".format(y)

    #     plt.annotate(label,  # this is the text
    #                   xy=(x, y), xytext=(-20,20), 
    #         textcoords='offset points', ha='center', va='bottom',
    #         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
    #         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
    #                         color='red')) 
   # plt.scatter(min_x, min_y, c="r" ,label="Minimum Capacity Costs")
    plt.legend(fontsize=16,loc='upper center')
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(1, 18, 1),fontsize=12)
    plt.xlabel("Number of control plane interfaces ($m$)", fontsize=18)
    plt.ylabel("Network  Capacity Costs [Mbps]", fontsize=18)
    plt.grid(axis = 'y')
    plt.tight_layout()
    plt.savefig(BASE_DIR + "/Figures/CostsContrib_Coronet30.png", format="png", pad_inches=0)


def plot_scaled_obj_val(f_name):

    solver_data=pd.read_excel(BASE_DIR + '/'+ f_name, sheet_name='Link_Utils',header=0)
    x_values=list(solver_data['Num_Nodes'])
    c_1 = solver_data[1]
    c_10 = solver_data[10]
    c_100 = solver_data[100]
    c_1000 = solver_data[1000]
    c_5000 = solver_data[5000]
    solver_data=pd.read_excel(BASE_DIR + '/'+ f_name, sheet_name='PL_NodeCosts',header=0)
    n_1 = solver_data['1.1']
    n_10 = solver_data['10.1']
    n_100 = solver_data['100.1']
    n_1000 = solver_data['1000.1']
    n_5000 = solver_data['5000.1']
    cc_1=c_1+n_1
    cc_10=c_10+n_10
    cc_100=c_100+n_100
    cc_1000=c_1000+n_1000
    cc_5000=c_5000+n_5000

    plt.figure(figsize=(10.5, 6.5), dpi=80)

    plt.plot(x_values, np.log10(cc_1), color="tab:blue", label="Scale_factor=1")
    plt.plot(x_values, np.log10(cc_10), color="tab:red", label="Scale_factor=10")
    plt.plot(x_values, np.log10(cc_100), color="tab:green", label="Scale_factor=100")
    plt.plot(x_values, np.log10(cc_1000), color="tab:brown", label="Scale_factor=1000")
    plt.plot(x_values, np.log10(cc_5000), color="tab:pink", label="Scale_factor=5000")
 # plot.yscale('log')
    # plot.yticks(np.arange(2000, 4000,250))



    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(0.7, 1.08))
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(1, 18, 1),fontsize=12)
    plt.xlabel("Number of control plane interfaces ($m$)", fontsize=18)
    plt.ylabel("Network  Capacity Costs [$log_{10}(C_{CP})$]", fontsize=18)
    plt.grid(axis = 'y')
    plt.tight_layout()
    plt.savefig(BASE_DIR + "/Figures/Scalefactor_Germany17_new.png", format="png", pad_inches=0)



    # min_x = x1[np.argmin(y1)]
    # min_y= min(y1)
    # plot.scatter(min_x, min_y,c='r', label='minimum')


def main():
    obj_fname='/Stats/Coronet30/Objectives.xlsx'
    plot_total_capacity_costs(obj_fname)
    plot_costs_contrib(obj_fname)
    networkfname='Stats/Euclid/Model_Stats_G17_M2_1.xlsx'
   # network_plot(networkfname, 'Hamburg')
    scale_fname='Stats/Euclid_New/Objectives_Scaled.xlsx'
    plot_scaled_obj_val(scale_fname)


if __name__ == "__main__":
    main()


