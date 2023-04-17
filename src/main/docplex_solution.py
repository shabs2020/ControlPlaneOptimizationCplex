#!/usr/bin/env python3
import cplex_input
from docplex.mp.model import Model
import numpy as np
import pandas as pd
import os
import write_workbook as wb
import matplotlib.pyplot as plot
import math
from itertools import combinations
import faulthandler

faulthandler.enable()
import sys
from datetime import datetime
import logging
import re


file_path = os.path.abspath(os.path.join(__file__, "../.."))
BASE_DIR = os.path.dirname(file_path)

logging.basicConfig(filename=BASE_DIR + "/Log/docplexlog_new_formulation_12.04_g17.txt", level=logging.INFO)

""" file_path = os.path.abspath(os.path.join(__file__ ,"../.."))
BASE_DIR = os.path.dirname(file_path)

# excel_file = BASE_DIR + '/Topologies/Coronet60.xlsx'
# node_file = BASE_DIR + '/Topologies/Nodes_Germany_17.json'
# edge_file = BASE_DIR + "/Topologies/Links_Germany_17.json"

node_file = BASE_DIR + '/Topologies/example_nodes.json'
edge_file = BASE_DIR + "/Topologies/example_links.json"
network = cplex_input.create_network(node_file, edge_file)

#network=cplex_input.create_network_from_excel(excel_file)
links = {}
i = 0
for e in network.edges:
    links[e] = ['e'+str(i), 1]
    i += 1

cplex_input.draw_topology(network, BASE_DIR + "/Figures/Germany_17") 

DIVERSITY_FACTOR = 2 """


def model_optimizer(
    network,
    links,
    volume_scale_factor,
    m,
    demand_volume,
    demand_paths,
    demand_path_edges,
    demand_path_lengths,
):
    ##############################################################################

    #                   create a CPLEX object

    ##############################################################################

    optimizer = Model(name="minimize cost of control plane network design")
    CPLEX_LP_PARAMETERS = {
        "benders.tolerances.feasibilitycut": 1e-9,
        "benders.tolerances.optimalitycut": 1e-9,
    }

    optimizer.parameters.barrier.algorithm = 3
    optimizer.parameters.optimalitytarget = 3
    optimizer.parameters.benders.tolerances.feasibilitycut = CPLEX_LP_PARAMETERS[
        "benders.tolerances.feasibilitycut"
    ]
    optimizer.parameters.benders.tolerances.optimalitycut = CPLEX_LP_PARAMETERS[
        "benders.tolerances.optimalitycut"
    ]

    ##############################################################################

    #                     Initialize input data

    ##############################################################################
    # direct_nodes, indirect_nodes = cplex_input.select_nodes(network, 5)

    set_links = [l[0] for l in links.values()]
    set_link_costs = [l[1] for l in links.values()]
    set_demands = [d for d in demand_volume]
    ###########################################################################################################################

    #                                        Parameter  - δedip

    #              Boolean parameter equals to 1 if if link e belongs to  potential path p for demand d;
    #                                        otherwise 0
    #                     Total number of this variable is: E * sum(d ∈ D) |P(d)|

    ###########################################################################################################################
    para_del_e_p = {}
    for e in set_links:
        for d in demand_path_edges:
            for p in demand_path_edges[d]:
                if e in demand_path_edges[d][p]:
                    para_del_e_p[e + "_" + d + "_" + p] = 1
                else:
                    para_del_e_p[e + "_" + d + "_" + p] = 0

    #########################################################################################################################################

    #                 Add variables

    # Note: CPLEX assigns each created variable a specific unique index, which allows an easy selection of the needed variable.

    # to add variables with CPLEX we need these arguments:
    # 1. objective - if this variable is used in the objective - normally is a list of ones whose length is equal to the number of variables
    # 2. lb - is the lower bound of the variables. It is usually 0.
    # 3. ub -is the upper bound of the variables. It is usually infinity.
    # 4. types - specify the type of variables. In our case all the variables are of type continuous
    # 5. names (optional) - specifies the names of variables

    ########################################################################################################################################

    ######################################################################################

    #                     Variable - mu_p^d

    #            binary variable if demand d is utilizing path p
    #              Total number of this variable D*sum P(d)

    ######################################################################################

    variable_name = []
    for d in demand_path_edges:
        variable_name.extend(["mu_" + d + "_" + p for p in demand_path_edges[d]])
    var_mu_d_p = optimizer.binary_var_dict(
        keys=variable_name, lb=0, ub=1, name=variable_name
    )
    ########################################################################################################################################

    ######################################################################################

    #                     Variable - x_d

    #            binary variable if demand d is of direct node then 1, else 0
    #              Total number of this variable N

    ######################################################################################

    variable_name = []

    variable_name.extend(["x_" + d for d in demand_volume.keys()])
    var_x_d = optimizer.binary_var_dict(
        keys=variable_name, lb=0, ub=1, name=variable_name
    )

    ###########################################################################################################################
    #            Constraint of equation 1: Node disjoint constraint
    #
    #                       for all  control demands d

    #           sum(p ∈ p(d->j))mu_p^d <= 1
    #           Total number of constraints of equation 1: |O| + |M|

    ###############################################################################
    for d in set_demands:
        for n in network.nodes:
            constraint_name = "c1_" + d + "_" + n
            paths_to_j = [i for i in demand_paths[d] if n == demand_paths[d][i][-1]]
            vars = [var_mu_d_p["mu_" + d + "_" + i] for i in paths_to_j]
            optimizer.add_constraint(
                ct=sum(vars) <= var_x_d["x_d_" + n], ctname=constraint_name
            )

    ###########################################################################################################################
    #            Constraint of equation 2: Link disjoint constraint
    #
    #            for all links e
    #                 for all demands d
    #                       sum(p ∈ p(d)) δedip*mu_p^d <= 1
    #           Total number of constraints of equation 1: |E|*|O|*|P(D)|

    ###############################################################################
    # sum(p ∈ p(d)) δedip*mu_p^d is also part of the objective function
    # hence part of code is used to deifne the objective kpi
    obj_kpi1 = []
    for e in set_links:
        for d in set_demands:
            constraint_name = "c2_" + e + "_" + d
            for p in demand_path_edges[d]:
                paras = [
                    para_del_e_p[e + "_" + d + "_" + p] for p in demand_path_edges[d]
                ]
                vars = [var_mu_d_p["mu_" + d + "_" + p] for p in demand_path_edges[d]]
            prod_vars_paras = np.multiply(paras, vars)
            obj_kpi1.extend(
                np.array(prod_vars_paras)
                * (demand_volume[d] / DIVERSITY_FACTOR)
                * set_link_costs[set_links.index(e)]
            )
            # print("length of kp1 {}".format(len(obj_kpi1)))
            optimizer.add_constraint(
                ct=sum(prod_vars_paras) <= 1 - var_x_d["x_" + d], ctname=constraint_name
            )
        # print(obj_kpi1)

    ###########################################################################################################################
    #            Constraint of equation 2: Path Diversity constraint
    #
    #                 for all demands d
    #                       sum(p ∈ p(d)) mu_p^d = n_d (Diversity Factor)
    #           Total number of constraints of equation 1: |O|

    ###############################################################################

    for d in set_demands:
        constraint_name = "c3_" + "_" + d
        vars = [var_mu_d_p["mu_" + d + "_" + p] for p in demand_path_edges[d]]
        optimizer.add_constraint(
            ct=sum(vars) == DIVERSITY_FACTOR * (1 - var_x_d["x_" + d]),
            ctname=constraint_name,
        )
    ###########################################################################################################################
    #            Constraint of equation 4: number of control plane interfaces
    #

    #                       sum(d ∈ N) x_d = m
    #           Total number of constraints of equation 1: |N|

    ################################################################################

    constraint_name = "c4_" + "_" + str(m)
    vars = [var_x_d["x_d_" + d] for d in network.nodes]
    optimizer.add_constraint(ct=sum(vars) == m, ctname=constraint_name)

    ###########################################################################################################################
    #         Objective Function
    #       F(M) = sum(e ∈ E) ξ_e * y_e + χ
    #            F(M)=   sum(e ∈ E) ξ_e *sum(d ∈ O) h_d/n_d  *sum(p ∈ p(d)) δedip*mu_p^d
    #                               + α* sum(d ∈ O) *sum(p ∈ p(d)) λ_p*mu_p^d

    ###############################################################################

    optimizer.add_kpi(sum(obj_kpi1), "link_util")
    obj_kp2 = []
    for d in set_demands:
        mu_p_d = [var_mu_d_p["mu_" + d + "_" + p] for p in demand_path_lengths[d]]
        path_length = [0.1 * demand_path_lengths[d][p] for p in demand_path_lengths[d]]
        obj_kp2.extend(np.multiply(mu_p_d, path_length))
    optimizer.add_kpi(sum(obj_kp2), "path_cost")

    optimizer.minimize(sum(obj_kpi1)+ sum(obj_kp2))
    return optimizer


def run_optimiser(
    network,
    links,
    scale_factor,
    demand_volume,
    demand_paths,
    demand_path_edges,
    demand_path_lengths,
):
    source_regex = r"mu_d_(.*)_"
    path_regex = r"mu_d_.*_(.*)"
    network_nodes = list(network.nodes)
    min_obj_per_M = {}
    kpi1_perf = {}
    kpi2_perf = {}
    total_episodes = len(network.nodes)
    print(total_episodes)
    ind_traffic_demand=[network.nodes[n]["demandVolume"] * scale_factor for n in network_nodes]

    final_node_costs=math.ceil(sum(ind_traffic_demand))
    avg_traffic_demand =cplex_input.round_capacity(sum(ind_traffic_demand)/len(network.nodes))
    for m in range(2, total_episodes + 1):

        
        if m == total_episodes:
            # orig_capacity = [
            #     network.nodes[n]["demandVolume"] * scale_factor for n in network_nodes
            # ]
            # capacity = [math.ceil(c) for c in orig_capacity]

            # print(capacity)
            # control_node_costs = sum(capacity)
            # min_obj_per_M[m] = control_node_costs
            control_node_costs = m*final_node_costs
        else:
            capacity = {}

            # Call Optimizer

            optimizer = model_optimizer(
                network,
                links,
                scale_factor,
                m,
                demand_volume,
                demand_paths,
                demand_path_edges,
                demand_path_lengths,
            )
            sol = optimizer.solve(log_output=True)
            if optimizer.solve_details.status == "integer optimal solution":
                variables_in_sol = sol.as_df()
                print(variables_in_sol)
                d_nodes = []
                for d in variables_in_sol["name"]:
                    if "mu_d" in d:
                        path_name = re.findall(path_regex, d, re.MULTILINE)[0]
                        print(path_name)
                        s_name = re.findall(source_regex, d, re.MULTILINE)[0]
                        print(s_name)

                        # print(network.nodes[demand_paths["d_" + s_name][path_name][-1]]["demandVolume"])
                        capacity[demand_paths["d_" + s_name][path_name][-1]] = (
                            capacity.get(
                                demand_paths["d_" + s_name][path_name][-1],
                                (network.nodes[
                                    demand_paths["d_" + s_name][path_name][-1]
                                ]["demandVolume"]
                                * scale_factor),
                            )
                            + demand_volume["d_" + s_name] / 2
                        )

                    else:
                        d_nodes.append(d[4:])
                remaining_direct_nodes = list(d_nodes - capacity.keys())
                for r in remaining_direct_nodes:

                    capacity[r] = network.nodes[r]["demandVolume"] * scale_factor
                for r in capacity:
                    capacity[r]=math.ceil(capacity[r])
                #total_capacity=[c for c in capacity.values()]
                #control_node_costs=sum(total_capacity)

                # print("Capacity per indirect node \n")
                control_node_costs = m*final_node_costs
                print(capacity)
                current_objective_value = sol.get_objective_value() + control_node_costs
                logging.info(
                    "Solution status is {}".format(optimizer.solve_details.status)
                )
                logging.info(
                    "Objective value for the solution is {}".format(
                        current_objective_value
                    )
                )
                logging.info("Number of variables {}".format(sol.number_of_var_values))
                print("Solution status is {}".format(optimizer.solve_details.status))
                print(
                    "Objective value for the solution is {}".format(
                        current_objective_value
                    )
                )
                print("Number of variables {}".format(sol.number_of_var_values))

                current_kp1 = sol.kpi_value_by_name("link_util")
                current_kp2 = sol.kpi_value_by_name("path_cost")

                logging.info("variables_in_sol {}".format(variables_in_sol))
                logging.info("Demand details {}".format(demand_volume))
                logging.info("demand_paths {}".format(demand_paths))

                # Write all input to excel
                kpi1_perf[m] = current_kp1
                kpi2_perf[m] = [current_kp2, control_node_costs]

                fname = r"/Stats/ICTONG17/Model_Stats_C301204_M" + str(m) + '_' + str(scale_factor)+ ".xlsx"

                #'_' + str(scale_factor)+
                book = wb.create_workbook(BASE_DIR + fname)
                book = wb.write_link_details(book, links)
                book = wb.write_demand_details(
                    book,
                    demand_volume,
                    demand_paths,
                    demand_path_edges,
                    demand_path_lengths,
                )
                book = wb.write_solution(book, variables_in_sol, "Solution_Variables")
                wb.save_book(book, BASE_DIR + fname)

            min_obj_per_M[m] = current_objective_value
    return min_obj_per_M, kpi1_perf, kpi2_perf


def plot_min_obj_value(f_name):
    obj_values = wb.load_workbook(f_name, "Obj_Values")
    x1 = obj_values[0]
    y1 = obj_values[1]
    plot.figure(figsize=(9, 4), dpi=80)
    color = "tab:blue"
    plot.plot(x1, y1, color=color)
    min_x = x1[np.argmin(y1)]
    min_y = min(y1)
    plot.scatter(min_x, min_y, c="r", label="minimum")
    plot.legend()
    plot.xticks(np.arange(1, 10, 1))
    # plot.yscale('log')
    # plot.yticks(np.arange(2000, 4000, 250))
    plot.xlabel("Number of Nodes", fontsize=18)
    plot.ylabel("Network Costs", fontsize=16)

    plot.tight_layout()
    plot.savefig(BASE_DIR + "/Figures/Minimumobjective.png", format="png", pad_inches=0)


def plot_scaled_obj_val(f_name, sheet_name, y_label, img_name):
    obj_values = wb.load_workbook(f_name, sheet_name)
    x_1 = obj_values[0]
    y_1 = obj_values[1]
    y_10 = obj_values[2]
    y_100 = obj_values[3]
    y_1000 = obj_values[4]
    y_5000 = obj_values[5]
    y_10k = obj_values[6]

    plot.figure(figsize=(9, 4), dpi=80)

    plot.plot(x_1, np.log(y_1), color="tab:blue", label="Scale_factor=1")
    plot.plot(x_1, np.log(y_10), color="tab:red", label="Scale_factor=10")
    plot.plot(x_1, np.log(y_100), color="tab:green", label="Scale_factor=100")
    plot.plot(x_1, np.log(y_1000), color="tab:brown", label="Scale_factor=1000")
    plot.plot(x_1, np.log(y_5000), color="tab:pink", label="Scale_factor=5000")
    plot.plot(x_1, np.log(y_10k), color="tab:cyan", label="Scale_factor=10000")
    # min_x = x1[np.argmin(y1)]
    # min_y= min(y1)
    # plot.scatter(min_x, min_y,c='r', label='minimum')
    plot.legend()
    plot.xticks(np.arange(1, 10, 1))
    # plot.yscale('log')
    # plot.yticks(np.arange(2000, 4000,250))
    plot.xlabel("Number of Nodes", fontsize=18)
    plot.ylabel(y_label, fontsize=16)

    plot.tight_layout()
    plot.savefig(img_name, format="png", pad_inches=0)


def run_sol_single():

    obj_record = BASE_DIR + "Stats/ICTONG17/Objectives_NewFormG171204.xlsx"
    scale_factor=1

    logging.info(
        "Time of Start : {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    )
    print("Time of Start: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    (
        demand_volume,
        demand_paths,
        demand_path_edges,
        demand_path_lengths,
    ) = cplex_input.define_control_demands(network, links, scale_factor)
    min_obj_per_M, kpi1_perf, kpi2_perf = run_optimiser(
        network,
        links,
        1,
        demand_volume,
        demand_paths,
        demand_path_edges,
        demand_path_lengths,
    )
    print(kpi1_perf)
    print(kpi2_perf)
    print(min_obj_per_M)
    book = wb.create_workbook(obj_record)
    book = wb.write_objective_values(book, min_obj_per_M, kpi1_perf, kpi2_perf)
    wb.save_book(book, obj_record)
    # plot_min_obj_value(obj_record)
    print(
        "Time of completetion: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    )
    logging.info(
        "Time of Completetion Traffic: {}".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    )


def run_sol_with_scale():
    obj_record = BASE_DIR + "/Stats/ICTONG17/ObjectivesG171204_Scaled.xlsx"
    SCALE_FACTORS = [1, 10, 100, 1000, 5000]
    min_obj_scaled = {}
    kpi_perf_scaled = {}
    kpi2_perf_scaled = {}

    for s_factor in SCALE_FACTORS:
        (
            demand_volume,
            demand_paths,
            demand_path_edges,
            demand_path_lengths,
        ) = cplex_input.define_control_demands(network, links, s_factor)
        logging.info(
            "Time of Start Scaled Traffic: {}".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        print(
            "Time of Start Scaled Traffic: {}".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        min_obj_per_M, kpi1_perf, kpi2_perf = run_optimiser(
            network,
            links,
            s_factor,
            demand_volume,
            demand_paths,
            demand_path_edges,
            demand_path_lengths,
        )
        min_obj_scaled[s_factor] = min_obj_per_M
        kpi_perf_scaled[s_factor] = kpi1_perf
        kpi2_perf_scaled[s_factor] = kpi2_perf
    print(min_obj_scaled)
    print(kpi_perf_scaled)
    print(kpi2_perf_scaled)
    book = wb.create_workbook(obj_record)
    book = wb.write_objective_values_scaled(book, min_obj_scaled, "Obj_Values")
    book = wb.write_objective_values_scaled(book, kpi_perf_scaled, "Link_Utils")
    book = wb.write_objective_values_scaled(book, kpi2_perf_scaled, "PL_NodeCosts")
    wb.save_book(book, obj_record)
    print(
        "Time of completetion: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    )
    logging.info(
        "Time of Completetion Scaled Traffice: {}".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    )

"""     plot_scaled_obj_val(obj_record, "Obj_Values", "Log(Network Costs)", img_name1)
    plot_scaled_obj_val(obj_record, "Link_Utils", "Link_Utilization (Y_e)", img_name2) """


def print_help():
    print("Cplex Optimzation Control Plane Design Start")
    print("=========================================")
    print("COMMANDS:")
    print("excelfile             give excel file name with topology details")
    print("nodefile              json file with nodes")
    print("edgefile              json file with edges")


if __name__ == "__main__":
    # sys.argv.pop(0)

    if len(sys.argv) < 1:
        print_help()
        exit(1)
    file_path = os.path.abspath(os.path.join(__file__, "../.."))
    BASE_DIR = os.path.dirname(file_path)
    excel_file, node_file, edge_file = "", "", ""

    execDirAbs = os.getcwd()
    sys.argv.pop(0)
    x = 0
    while x < len(sys.argv):
        arg = sys.argv[x]
        print(arg)
        if arg == "excelfile":
            sys.argv.pop(x)
            arg = sys.argv[x]
            sys.argv.pop(x)
            excel_file = BASE_DIR + "/" + arg
            print("Excel file found")
        elif arg == "nodefile":
            sys.argv.pop(x)
            arg = sys.argv[x]
            node_file = BASE_DIR + "/" + arg
            sys.argv.pop(x)
        elif arg == "edgefile":
            sys.argv.pop(x)
            arg = sys.argv[x]
            edge_file = BASE_DIR + "/" + arg
            sys.argv.pop(x)
        # elif arg == "csvfile":
        #     sys.argv.pop(x)
        #     arg = sys.argv[x]
        #     sys.argv.pop(x)
        #     csv_file = BASE_DIR + "/" + arg
        #     print("Csv file found")

    if excel_file:
        network = cplex_input.create_network_from_excel(excel_file)
        print("Network created")
    # elif csv_file:
    #     network = cplex_input.create_network_from_csv(csv_file)
    #     print("Network created")
    elif node_file and edge_file:
        network = cplex_input.create_network(node_file, edge_file)
        print("Network created")
    else:
        print("Node or Edge Information missing")
        exit(1)

    links = {}
    i = 0
    for e in network.edges:
        links[e] = ["e" + str(i), network.edges[e]["linkCost"]]
        i += 1
    DIVERSITY_FACTOR = 2
   # run_sol_single()
    run_sol_with_scale()


# run_sol_single()
# run_sol_with_scale()

# min_obj_per_M, kpi1_perf = run_optimiser(network, links, 1)
# print(min_obj_per_M)
# sol = optimizer.solve(log_output=True)
# variables_in_sol = sol.as_df()
# fname = r'/Model_Stats_new.xlsx'
# book = wb.create_workbook(BASE_DIR+fname)
# book = wb.create_workbook(BASE_DIR+fname)

# book = wb.write_link_details(book, links)
# book = wb.write_demand_details(
#     book, demand_volume, demand_paths, demand_path_edges, demand_path_lengths)
# book = wb.write_solution(book, variables_in_sol)
# wb.save_book(book, BASE_DIR+fname)
# print("Solution status is {}".format(optimizer.solve_details.status))
# print("Objective value for the solution is {}".format(sol.get_objective_value()))
# print(sol)


# obj_episode={}
# min_obj_per_M={}
# while(M>=2):
#     obj_per_epoch, min_objective_value,min_obj_per_M,total_stats = run_optimiser(network, links,100,M)
#     obj_episode[M] =[obj_per_epoch]
#     min_obj_per_M[M]= total_stats
#     M=M-1
# print(min_obj_per_M)
# x1=[]
# y1=[]
# node_costs=[]
# capacity_costs=[]
# for m in min_obj_per_M:
#     x1.append(m)
#     y1. append(min_obj_per_M[m][0])
#     node_costs.append(min_obj_per_M[m][1])
#     capacity_costs.append(min_obj_per_M[m][2])

# x1.reverse()
# y1.reverse()
# node_costs.reverse()
# capacity_costs.reverse()
# print(node_costs)
# print(capacity_costs)
# plot.figure(figsize=(9, 4), dpi=80)
# plot.plot(x1,y1)
# plot.xticks(np.arange(1,9))
# plot.yticks(np.arange(0, 250,30))
# plot.xlabel("Number of Control Nodes", fontsize=18)
# plot.ylabel('Objective Value', fontsize=16)
# plot.tight_layout()
# plot.savefig(BASE_DIR+"/DesignObjective.png", format='png', pad_inches=0)
# plot.clf()
# fig, ax1 = plot.subplots(figsize=(10, 5.3))
#     # figsize=(40, 30)
# color = 'tab:blue'
# ax1.set_xlabel('Number of Control Nodes', fontsize=20)
# ax1.set_ylabel('Capacity Costs', color=color, fontsize=20)
# ax1.plot(x1, capacity_costs, color=color,linewidth=0.75)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:orange'

# ax2.set_ylabel('Control Node Costs', color=color, fontsize=20)  # we already handled the x-label with ax1
# ax2.plot(x1, node_costs, color=color, linewidth=0.75)
# ax2.tick_params(axis='y', labelcolor=color)

# #  ax1.set_xlim(0, 51)
# ax1.set_ylim(0,150)
# ax2.set_ylim(0,150)
# ax1.yaxis.labelpad = 0
# ax2.yaxis.labelpad = 0

# plot.tight_layout()
# plot.savefig(BASE_DIR+'/Cost_Comaprison' + ".png")
# # bbox_inches='tight'
# plot.clf()
