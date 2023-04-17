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

logging.basicConfig(filename=BASE_DIR + "/Log/docplexlog_ictonG17.txt", level=logging.INFO)

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
    cc_paths,cc_paths_edges,cc_path_lengths

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
    for e,d in ((e,d) for e in set_links for d in demand_path_edges):
        for p in demand_path_edges[d]:
            if e in demand_path_edges[d][p]:
                para_del_e_p[e + "_" + d + "_" + p] = 1
            else:
                para_del_e_p[e + "_" + d + "_" + p] = 0

    #########################################################################################################################################

  ###########################################################################################################################

    #                                        Parameter  - δedip

    #              Boolean parameter equals to 1 if if link e belongs to  potential path p for demand d;
    #                                        otherwise 0
    #                     Total number of this variable is: E * sum(d ∈ D) |P(d)|

    ###########################################################################################################################
    para_del_e_i_j = {}
    for e,p in ((e,p) for e in set_links for p in cc_paths_edges):
        for path in cc_paths_edges[p]:
            if e in cc_paths_edges[p][path]:
                para_del_e_i_j[e + "_" + p[0]+'/'+p[1] + "_" + path] = 1
            else:
                para_del_e_i_j[e + "_" + p[0]+'/'+p[1] + "_" + path] = 0
    #print(para_del_e_i_j)
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
    
    variable_name.extend(["mu_" + d + "_" + p for d in demand_path_edges for p in demand_path_edges[d]])
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
    ######################################################################################


    #                     Variable - mu_p^nn'

    #            binary variable if node  is utilizing path p

    #              Total number of this variable D*sum P(d)

    ######################################################################################

    variable_name = []
    
    variable_name.extend(["mu_"+d[0]+"/"+d[1] + "_" + p for d in cc_paths for p in cc_paths[d]])
    var_mu_j_p = optimizer.binary_var_dict(
        keys=variable_name, lb=0, ub=1, name=variable_name
    )

    ########################################################################################################################################

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

            paras = [
                para_del_e_p[e + "_" + d + "_" + p] for p in demand_path_edges[d]
            ]
            vars = [var_mu_d_p["mu_" + d + "_" + p] for p in demand_path_edges[d]]
            prod_vars_paras = np.multiply(paras, vars)
            obj_kpi1.extend(
                np.array(prod_vars_paras)

                * demand_volume[d]

                * (demand_volume[d] / DIVERSITY_FACTOR)

                * set_link_costs[set_links.index(e)]
            )
            # print("length of kp1 {}".format(len(obj_kpi1)))
            optimizer.add_constraint(
                ct=sum(prod_vars_paras) <= 1 - var_x_d["x_" + d], ctname=constraint_name
            )
        # print(obj_kpi1)

    ###########################################################################################################################
    #            Constraint of equation 3: Path Diversity constraint
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
    #            Controller to controller connection
 
   ###########################################################################################################################

    #            Constraint of equation 5: interconnection only when each node in node pair is direct
    #
    #                      for all node pairs (n,n')

    #           sum(p ∈ P(n,n'))mu_p^(n,n') <= x_n
    #           sum(p ∈ P(n,n'))mu_p^(n,n') <= x_n'
    #           Total number of constraints of equation 1: N(N-1)

    ###############################################################################
    for p in cc_paths:
        constraint_name_i="c1_j_" + p[0] 
        constraint_name_j="c1_j_" + p[1] 
        vars = [var_mu_j_p["mu_"+p[0]+"/"+p[1] + "_" + q] for q in cc_paths[p]]
        optimizer.add_constraint(ct=sum(vars)<=var_x_d["x_d_" + p[0]],ctname=constraint_name_i)
        optimizer.add_constraint(ct=sum(vars)<=var_x_d["x_d_" + p[1]],ctname=constraint_name_j)
   ###########################################################################################################################
    #            Constraint of equation 6: each node is connected to two other nodes
    #
    #                      for all nodes n:
    #           sum(p ∈ P(n,n'))mu_p^(n,n') + sum(p ∈ P(n',n))mu_p^(n',n) >= eta*x_n
    #           Total number of constraints of equation 1: N*N(N-1)

    ############################################################################### 

    for n in network.nodes:
        vars = [var_mu_j_p[f"mu_{p[0]}/{p[1]}_{paths}"] for p in cc_paths for paths in cc_paths[p] if n in p]
        constraint_name='c2_j' + n
        optimizer.add_constraint(ct=sum(vars) >= var_x_d[f'x_d_{n}'] * DIVERSITY_FACTOR,ctname=constraint_name)

      
   ###########################################################################################################################
    #       Constraint of equation 7: Link disjoint
    #       for all links e:
    #            for all nodes n:
    #           sum(p ∈ P(n,n')) δeij*mu_p^(n,n') + sum(p ∈ P(n',n)) δedip*mu_p^(n',n)<=x_n
    #           Total number of constraints of equation 1: E*N*N(N-1)

    ###############################################################################
    prod_vars_paras_j=[0]
    prod_vars_paras_i=[0]

    for e in set_links:
        for n in network.nodes:
            vars1=[var_mu_j_p[f"mu_{p[0]}/{p[1]}_{paths}" ] for p in cc_paths_edges for paths in cc_paths_edges[p] if n in p]
            paras1=[para_del_e_i_j[e + "_" + p[0]+'/'+p[1] + "_" + paths]for p in cc_paths_edges for paths in cc_paths_edges[p] if n in p]
            prod_vars_paras=np.multiply(vars1,paras1)
            if n=='Hamburg':
                print(f'vars1{vars1} for node {n} and link {e}')
                print(f'paras1{paras1} for node {n} and link {e}')

                    #print(var_mu_j_p)
                    # print(f'vars1{vars1} for node {n} and link {e}')
                    # print(f'paras1{paras1}')
                    # print(f'prod_vars_paras{prod_vars_paras} for {e}')
            constraint_name='c3_j' + n
            optimizer.add_constraint(ct=sum(prod_vars_paras)<= 1,ctname=constraint_name)
        


 
    optimizer.add_kpi(sum(obj_kpi1), "link_util")
    obj_kp2 = []
    for n in network.nodes:
        for p in cc_path_lengths:
            mu_p_d = [var_mu_j_p[f"mu_{p[0]}/{p[1]}_{paths}" ] for paths in cc_path_lengths[p] if n in p]
            path_length = [cc_path_lengths[p][path] for path in cc_path_lengths[p] if n in p]
            obj_kp2.extend(np.multiply(mu_p_d, path_length))
    optimizer.add_kpi(sum(obj_kp2), "controller_path_cost")
    optimizer.minimize_static_lex([sum(obj_kpi1), sum(obj_kp2)])
    return optimizer

def model_optimiser_controller(cc_paths:dict, cc_path_edges:dict, cc_path_lengths:dict):
    set_links = [l[0] for l in links.values()]
    ##############################################################################

    #                   create a CPLEX object

    ##############################################################################

    optimizer = Model(name="optmize routing of controller-to-controller conn.")
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
    ###########################################################################################################################

    #                                        Parameter  - δedip

    #              Boolean parameter equals to 1 if if link e belongs to  potential path p for demand d;
    #                                        otherwise 0
    #                     Total number of this variable is: E * sum(d ∈ D) |P(d)|

    ###########################################################################################################################
    para_del_e_p = {}
    for e,j in ((e,j) for e in set_links for j in cc_path_edges):
        for p in cc_path_edges[j]:
            if e in cc_path_edges[j][p]:
                para_del_e_p[e + "_" + j + "_" + p] = 1
            else:
                para_del_e_p[e + "_" + j + "_" + p] = 0
     ######################################################################################

    #                     Variable - mu_p^j

    #            binary variable if demand d is utilizing path p
    #              Total number of this variable D*sum P(d)

    ######################################################################################

    variable_name = []
    
    variable_name.extend(["mu_" + d + "_" + p for d in cc_paths for p in cc_paths[d]])
    var_mu_d_p = optimizer.binary_var_dict(
        keys=variable_name, lb=0, ub=1, name=variable_name
    )
    ########################################################################################################################################

    ###########################################################################################################################
    #            Constraint of equation 1: Node disjoint constraint
    #
    #            for all  direct nodes j

    #           sum(p ∈ p(j->j'))mu_p^j <= 1
    #           Total number of constraints of equation 1: |M| + |M|

    ###############################################################################

    for j,k in ((j,k) for j in cc_paths.keys() for k in cc_paths.keys() if j!=k):
        constraint_name = "c1_" + j + "_" + k
        paths_j_to_k=[path for path in cc_paths[j] if k==cc_paths[j][path][-1]]
        vars = [var_mu_d_p["mu_" + j + "_" + i] for i in paths_j_to_k]
        optimizer.add_constraint(ct=sum(vars)<=1,ctname=constraint_name)
    ###########################################################################################################################
    #            Constraint of equation 2: Path Diversity constraint
    #
  #            for all  direct nodes j
    #                       sum(p ∈ p(j)) mu_p^j = n_d (Diversity Factor)
    #           Total number of constraints of equation 1: |M|

    ###############################################################################

    for j in cc_path_edges:
        constraint_name = "c3_" + "_" + j
        vars = [var_mu_d_p["mu_" + j + "_" + p] for p in cc_path_edges[j]]
        optimizer.add_constraint(
            ct=sum(vars) == 2 ,
            ctname=constraint_name
        )
     ###########################################################################################################################
    #            Constraint of equation 3: Link disjoint constraint
    #
    #            for all links e
    #                 for all controllers j
    #                       sum(p ∈ p(j)) δedip*mu_p^j <= 1
    #           Total number of constraints of equation 1: |E|*|M|*|P(D)|

    ###############################################################################
    # sum(p ∈ p(d)) δedip*mu_p^d is also part of the objective function
    # hence part of code is used to deifne the objective kpi
    obj_kpi1 = []

    for e,j in ((e,j) for e in set_links for j in cc_path_edges):
        constraint_name = "c2_" + e + "_" + j
        paras = [para_del_e_p[e + "_" + j + "_" + p] for p in cc_path_edges[j]]
        vars = [var_mu_d_p["mu_" + j + "_" + p] for p in cc_path_edges[j]]
        optimizer.add_constraint(
                ct=sum(np.multiply(paras, vars)) <= 1, ctname=constraint_name
            )

   ###########################################################################################################################
    #         Objective Function

    #            F(M)=   sum(d ∈ O) *sum(p ∈ p(d)) λ_p*mu_p^d

    ###############################################################################

    obj_kp2=[np.dot([var_mu_d_p["mu_" + j + "_" + p] for p in cc_path_lengths[j]], [cc_path_lengths[j][p] for p in cc_path_lengths[j]]) for j in cc_path_lengths]
    optimizer.add_kpi(sum(obj_kp2), "path_cost")
    optimizer.minimize(sum(obj_kp2))
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


    total_episodes = len(network.nodes)
    print(total_episodes)
    for m in range(3, 4):

        if m == total_episodes:
            orig_capacity = [
                network.nodes[n]["demandVolume"] * scale_factor for n in network_nodes
            ]
            capacity = [math.ceil(c) for c in orig_capacity]

            print(capacity)
            control_node_costs = sum(capacity)
            min_obj_per_M[m] = control_node_costs
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

                cc_paths,cc_paths_edges,cc_path_lengths
            )
            sol = optimizer.solve(log_output=True)
            variables_in_sol = sol.as_df()
                
            if optimizer.solve_details.status == "multi-objective optimal":
                variables_in_sol = sol.as_df()
                print(variables_in_sol)

                # print("Capacity per indirect node \n")
                print(capacity)
                current_objective_value = sol.get_objective_value()

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


                logging.info("variables_in_sol {}".format(variables_in_sol))
                logging.info("Demand details {}".format(demand_volume))
                logging.info("demand_paths {}".format(demand_paths))



                fname = r"/Stats/Stats/Icton/Model_Stats_ICTON_M" + str(m) + '_' + str(scale_factor)+ ".xlsx"


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

                book = wb.write_nodepair_paths(book, cc_paths,cc_path_lengths,cc_paths_edges)
                book = wb.write_solution(book, variables_in_sol,"Solution_Variables")
                wb.save_book(book, BASE_DIR + fname)
                min_obj_per_M[m] = current_objective_value
    return min_obj_per_M



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


    obj_record = BASE_DIR + "/Stats/Stats/Icton/Objectives_ICTONG17.xlsx"

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

    min_obj_per_M = run_optimiser(

        network,
        links,
        1,
        demand_volume,
        demand_paths,
        demand_path_edges,
        demand_path_lengths,
    )


    print(min_obj_per_M)
    book = wb.create_workbook(obj_record)
    book = wb.write_multi_objective_values(book, min_obj_per_M)

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
    obj_record = BASE_DIR + "/Stats/NewFormulation/Objectives_Scaled.xlsx"
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
        elif arg == "csvfile":
            sys.argv.pop(x)
            arg = sys.argv[x]
            sys.argv.pop(x)
            csv_file = BASE_DIR + "/" + arg
            print("Csv file found")


    if excel_file:
        network = cplex_input.create_network_from_excel(excel_file)
        print("Network created")
    elif node_file and edge_file:
        network = cplex_input.create_network(node_file, edge_file)
        print("Network created")
    elif csv_file:
        network = cplex_input.create_network_from_csv(csv_file)
        print("Network created")
    else:
        print("Node or Edge Information missing")
        exit(1)

    links={e: ['e'+ str(i), network.edges[e]['linkCost']] for i,e in enumerate(network.edges) }
    cc_paths,cc_paths_edges,cc_path_lengths=cplex_input.find_all_node_pair_paths(network,links)

    DIVERSITY_FACTOR = 2
    run_sol_single()
    # run_sol_with_scale()


