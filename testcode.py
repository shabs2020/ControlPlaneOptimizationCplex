import cplex_input
import write_workbook as wb
import os

file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(file_path)

node_file = BASE_DIR + '/example_nodes.json'
edge_file = BASE_DIR + "/example_links.json"
network = cplex_input.create_network(node_file, edge_file)


#def model_optimizer(network):

links = {}
i = 0
for e in network.edges:
    links[e] = ['e'+str(i), 1]
    i += 1
direct_nodes = cplex_input.select_direct_nodes(network=network)
indirect_nodes = list(set(list(network.nodes)).difference(direct_nodes))

demands_dict = cplex_input.define_control_demands(
    indirect_nodes, direct_nodes, links, network)
set_demands = [d[0] for d in demands_dict.values()]
set_links = [l[0] for l in links.values()]
unit_linkcost = [l[1] for l in links.values()]

print(demands_dict)
fname = r'/Model_Stats.xlsx'
book = wb.create_workbook(BASE_DIR+fname)

book = wb.write_link_details(book,links)
book = wb.write_demand_details(book, demands_dict)
wb.save_book(book,BASE_DIR+fname)