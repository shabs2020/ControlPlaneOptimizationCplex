import cplex_input
import write_workbook as wb
import os

file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(file_path)

# node_file = BASE_DIR + '/example_nodes.json'
# edge_file = BASE_DIR + "/example_links.json"
# network = cplex_input.create_network(node_file, edge_file)


# #def model_optimizer(network):

# links = {}
# i = 0
# for e in network.edges:
#     links[e] = ['e'+str(i), 1]
#     i += 1
# direct_nodes = ['Berlin', 'Dortmund']
# indirect_nodes = list(
#             set(list(network.nodes)).difference(direct_nodes))

# demands_dict = cplex_input.define_control_demands(
#     indirect_nodes, direct_nodes, links, network)
# set_demands = [d[0] for d in demands_dict.values()]
# set_links = [l[0] for l in links.values()]
# unit_linkcost = [l[1] for l in links.values()]

# print(demands_dict)
# fname = r'/Model_Stats_test.xlsx'
# book = wb.create_workbook(BASE_DIR+fname)

# book = wb.write_link_details(book,links)
# book = wb.write_demand_details(book, demands_dict)
# wb.save_book(book,BASE_DIR+fname)


# min_obj_saled={1: {2: 2917.4849999999997, 3: 2447.0200000000004, 4: 2445.84, 5: 2535.815, 6: 2653.855, 7: 2926.375, 8: 3245.16, 9: 3600}, 10: {2: 4217.78, 3: 3166.8849999999998, 4: 2911.6349999999998, 5: 2916.92, 6: 2950.27, 7: 3138.1, 8: 3329.85, 9: 3600}, 100: {2: 17133.004999999997, 3: 10365.535, 4: 7569.585, 5: 6490.985000000001, 6: 5649.370000000001, 7: 4835.82, 8: 4176.75, 9: 3600}, 1000: {2: 146285.25499999995, 3: 82352.03499999997, 4: 54149.08499999999, 5: 40366.98499999999, 6: 31056.36999999999, 7: 21773.819999999996, 8: 12645.749999999998, 9: 3600}, 5000: {2: 720295.2549999999, 3: 402292.0349999999, 4: 261169.08499999996, 5: 190926.98499999996, 6: 143976.37, 7: 97053.81999999998, 8: 50285.74999999999, 9: 3600}, 10000: {2: 1437807.755, 3: 802217.0349999998, 4: 519944.0849999999, 5: 379126.9849999999, 6: 285126.37, 7: 191153.81999999998, 8: 97335.74999999999, 9: 3600}}
# book=wb.create_workbook(BASE_DIR + '/Stats/Scaled_Objective_Vals')

import os
print('PYTHONPATH="%s"' % os.environ['PYHTHONPATH'])
