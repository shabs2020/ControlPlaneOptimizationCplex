import cplex_input
import cplex
from docplex.mp.model import Model
import numpy as np
import pandas as pd
import os
import write_workbook as wb
import matplotlib.pyplot as plot

file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(file_path)


node_file = BASE_DIR + '/example_nodes.json'
edge_file = BASE_DIR + "/example_links.json"
network = cplex_input.create_network(node_file, edge_file)
links = {}
i = 0
for e in network.edges:
    links[e] = ['e'+str(i), 1]
    i += 1


def model_optimizer(network, links, direct_nodes, indirect_nodes):

    ##############################################################################

    #                   create a CPLEX object

    ##############################################################################

    optimizer = Model(name="minimize cost of control plane network design")
    CPLEX_LP_PARAMETERS = {
        'benders.tolerances.feasibilitycut': 1e-9,
        'benders.tolerances.optimalitycut': 1e-9
    }

    optimizer.parameters.barrier.algorithm = 3
    optimizer.parameters.optimalitytarget = 3
    optimizer.parameters.benders.tolerances.feasibilitycut = CPLEX_LP_PARAMETERS[
        'benders.tolerances.feasibilitycut']
    optimizer.parameters.benders.tolerances.optimalitycut = CPLEX_LP_PARAMETERS[
        'benders.tolerances.optimalitycut']

    ##############################################################################

    #                     Initialize input data

    ##############################################################################

    demands_dict = cplex_input.define_control_demands(
        indirect_nodes, direct_nodes, links, network)
    set_demands = [d[0] for d in demands_dict.values()]
    set_links = [l[0] for l in links.values()]
    unit_linkcost = [l[1] for l in links.values()]

    all_paths = {}
    for d in demands_dict:
        for i in demands_dict[d][2]:
            all_paths[i[0]] = demands_dict[d][2][i]

    path_per_link = {}
    for e in set_links:
        path_per_link[e] = [k for k in all_paths if e in all_paths[k]]

    set_paths = [k for k in all_paths]

    N_direct_nodes = len(direct_nodes)
    N_indirect_nodes = len(indirect_nodes)
    N_links = len(set_links)
    N_paths = len(set_paths)
    # N_demands = len(set_demands)

    # Defining the u_j value. by default it is 0
    u_ij = {}
    names_i_to_j = [
        'u' + i + j for i in indirect_nodes for j in direct_nodes]
    for n in names_i_to_j:
        u_ij[n] = 0
    # set the value of u_ij based on the calculated paths
    for i in indirect_nodes:
        paths = [l for l in demands_dict[i][2]]
        # print(i)
        # print(paths)

        for j in direct_nodes:
            node_exists = [(x, y) for x, y in paths if y == j]
            if len(node_exists) > 0:
                u_ij['u'+i+j] = 1

    # set the objective function
    optimizer.set_objective_sense(optimizer.objective_sense.Minimize)


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

    #                     Variable - Ye0

    #            capacity of IP link e ∈ E in failure free mode

    #                forall IP links e ∈ E:  Ye0 >= 0
    #            Total number of this variable is equal to the number of IP links.

    ######################################################################################

    varnames_link_capacity = ["y" + str(e) for e in set_links]
    objective_coeff = []

    var_link_capacity = optimizer.continuous_var_dict(keys=N_links, lb=0.0,
                                                      ub=cplex.infinity, name=varnames_link_capacity)

    ###########################################################################################################################

    #                                        Variable  - xdp0

    #              Nominal capacity of tunnel for demand d ∈ D routed over path p ∈ P(d) in failure free mode
    #                 forall demands d ∈ D, forall paths p ∈ p(d):    xdp0 >= 0
    #                      Total number of this variable is: sum(d ∈ D) |P(d)|

    ###########################################################################################################################

    varnames_demandvol_per_path = []
    for d in demands_dict.values():
        xname = ['x' + d[0] + p[0][0] for p in d[2].items()]
        varnames_demandvol_per_path.extend(xname)
    N_total_paths = len(varnames_demandvol_per_path)

    var_xdp = optimizer.continuous_var_dict(
        keys=N_total_paths, lb=0.0, ub=cplex.infinity, name=varnames_demandvol_per_path)

    ###########################################################################################################################

    #                                        Variable  - uij

    #              Boolean variable equals to 1 if indirect node i is connected to direct node j
    #                     Total number of this variable is: N * M

    ###########################################################################################################################

    # varnames_i_to_j = [
    #     'u' + i + j for i in indirect_nodes for j in direct_nodes]
    # var_uij = optimizer.integer_var_dict(lb=0, ub=1,
    #                                     keys=N_direct_nodes*N_indirect_nodes, name=varnames_i_to_j)
    ###########################################################################################################################

    #                                        Variable  - δedip

    #              Boolean variable equals to 1 if if link e belongs to path p realizing demand di of node i;
    #                                        otherwise 0
    #                     Total number of this variable is: E * sum(d ∈ D) |P(d)|

    ###########################################################################################################################
    # varnames_link_inpath = []
    # for e in set_links:

    #     for d in demands_dict.values():
    #         xname = [e + d[0] + p[0][0] for p in d[2].items()]
    #         varnames_link_inpath.extend(xname)
    # var_del_di_p = optimizer.integer_var_dict(lb=0, ub=1,
    #                                         keys=len(varnames_link_inpath), name=varnames_link_inpath)

    ###########################################################################################################################

    #                                        Variable  - M
    # Integer Variable M
    

    ###########################################################################################################################
    
    var_M=optimizer.integer_var(lb=2, ub=len(network.nodes), name='M')

    #            Constraint of equation 1: Capacity constraint
    #
    #                       for all  links e ∈ E:

    #           sum(d ∈ D) sum(p ∈ p(d)) δedip*xdip - Ye <= 0
    #           Total number of constraints of equation 1: |E|

    ###############################################################################

    for e in range(N_links):
        constraint_name_1 = 'c1_e' + str(e+1)
        # right hand side is a list of zeros since the variables of type one we can bring them to the left hand side
        rhs1 = [0.0]
        # In newstr we save the numbers of the elements of the set of Paths per link - for example: 0,3,4,7,9,11
        newstr = ''.join((ch if ch in '0123456789.-e' else ' ')
                         for p in path_per_link['e'+str(e)] for ch in p)
        listOfNumbers = [int(i) for i in newstr.split()]

        ind = [var_xdp[p] for p in listOfNumbers]
        [var_link_capacity[e]]

        # val gives the coefficients of the indicies, length of the list of ones is the same as the length of listOfNumbers
    # val = [var_del_di_p[e*N_paths + x] for x in listOfNumbers]
        optimizer.add_constraint(ct=sum(ind[i] for i in range(
            len(ind)))-var_link_capacity[e] <= 0.0, ctname=constraint_name_1)

        # try:
        #     if len(ind) == len(val):
        #         var_sum = np.dot(ind,val)

        #         # optimizer.add(var_sum<=var_link_capacity[e])
        #         # quadractic_expr.append(var_sum<=var_link_capacity[e])
        #         # print(quadractic_expr)

        #         # optimizer.add_constraint(ct=var_link_capacity[e]>=0)
        #        # optimizer.add_constraint(ct=0<=np.prod(val)<=1)
        #         optimizer.add_constraint(ct=sum(ind[i])-var_link_capacity[e]<=0.0, ctname=constraint_name_1)

        #     else:
        #         raise ValueError(
        #             "Length of variables and coefficients do not match")
        # except ValueError as exp:
        #     print("Error", exp)
    # optimizer.add_quadratic_constraints(quadractic_expr)

    ######################################################################################

    #               Constraint of type 2: Flow conservation constraint

    #                 forall demands d ∈ D :

    #                    sum(p ∈ p(d)) xdp = h(d)
    #                 Total number of constraints of type 2: |D|

    ######################################################################################

    for d in demands_dict:
        constraint_name_2 = 'c2_' + demands_dict[d][0]
        paths_per_demand = [i[0] for i in demands_dict[d][2]]
        # print(paths_per_demand)
        newstr = ''.join((ch if ch in '0123456789.-e' else ' ')
                         for p_d in paths_per_demand for ch in p_d)
        listOfNumbers = [int(i) for i in newstr.split()]
        ind = [var_xdp[p] for p in listOfNumbers]
        val = [1.0] * len(listOfNumbers)
        var_sum = optimizer.sum(ind[i]*val[i]
                                for i in range(len(listOfNumbers)))
        optimizer.add_constraint(ct=var_sum == float(
            demands_dict[d][1]), ctname=constraint_name_2)

    ######################################################################################

    #               Constraint of type 3: Node disjoint constraint

    #                 forall indirect nodes i :

    #                    sum(j ∈ M) u_ij >= 2
    #                 Total number of constraints of type 2: |N|

    ######################################################################################

    """ for i in range(len(indirect_nodes)):
        constraint_name_3 = 'c_u_' + indirect_nodes[i]

        listOfNumbers = [i*len(direct_nodes)+j for j in range(len(direct_nodes))]
        ind = [var_uij[l] for l in listOfNumbers]
        #val = [1.0] * len(listOfNumbers)
        var_sum = optimizer.sum(ind[i] for i in range(len(listOfNumbers)))
        optimizer.add_constraint(ct=var_sum >= 2, ctname=constraint_name_3) """

    ######################################################################################

    #               Constraint of type 4: Path diversity constraint

    #                 for all demands di :
    #                      x_dip-h(d)/sum(u_ij) <=0
    #                       sum(u_ij) can be replaced as the number of paths

    #                    x_dip<=h(d)/N(P(d_i))
    #                 Total number of constraints of type 2: |D|*P(d)

    ######################################################################################

    for d in demands_dict:

        paths_per_demand = [i[0] for i in demands_dict[d][2]]
        rhs = float(demands_dict[d][1])/len(paths_per_demand)
        # print(paths_per_demand)
        for p in paths_per_demand:
            constraint_name_4 = 'c4_' + demands_dict[d][0] + p
            newstr = ''.join((ch if ch in '0123456789.-e' else ' ')
                             for ch in p)

            listOfNumbers = int(newstr.split()[0])

            ind = var_xdp[listOfNumbers]

            optimizer.add_constraint(ct=ind <= rhs, ctname=constraint_name_4)

    ######################################################################################

    #               Constraint of type 5: Link Disjoint constraint

    #                 for all links e :
    #                       δedip<=1

    #                 Total number of constraints of type 2: |E|*|D|

    ######################################################################################

    # for e in range(N_links):
    #     for d in demands_dict:
    #         constraint_name_5 = 'c5_e' + str(e) + demands_dict[d][0]
    #         paths_per_demand = [i[0] for i in demands_dict[d][2]]
    #         newstr = ''.join((ch if ch in '0123456789.-e' else ' ')
    #                         for p_d in paths_per_demand for ch in p_d)
    #         listOfNumbers = [int(i) for i in newstr.split()]
    #         ind = [var_del_di_p[e*N_paths + i] for i in listOfNumbers]
    #         #val = [1.0] * len(listOfNumbers)

    #         var_sum = optimizer.sum(ind[i] for i in range(len(listOfNumbers)))
    #         optimizer.add_constraint(var_sum <= 1, ctname=constraint_name_5)

    keys = list(demands_dict.keys())
    u_ij_var = []
    link_var = []
    node_costs=0
    for i in indirect_nodes:
        paths = [l for l in demands_dict[i][2]]
        node_costs=demands_dict[i][3]+node_costs
        for j in paths:
            newstr = ''.join((ch if ch in '0123456789' else ' ')
                             for e in demands_dict[i][2][j] for ch in e)
            listOfNumbers = [int(i) for i in newstr.split()]
            sum_link_var = optimizer.sum(
                var_link_capacity[e] for e in listOfNumbers)*u_ij['u' + i + j[1]]
            print(sum_link_var)
            link_var.append(sum_link_var)

    capacity_costs=sum(link_var)+node_costs
    optimizer.add_kpi(capacity_costs)
    
    optimizer.minimize(sum(link_var)+node_costs)

    return optimizer, var_link_capacity, var_xdp, demands_dict


def run_optimiser(network, links, num_epochs,M):
    min_objective_value = 10000.00
    obj_per_epoch={}
    min_obj_per_epoch={}
    
    for epoch in range(num_epochs):
        direct_nodes = cplex_input.select_direct_nodes(network=network,M=M)
        indirect_nodes = list(
            set(list(network.nodes)).difference(direct_nodes))

        optimizer, var_link_capacity, var_xdp,demands_dict= model_optimizer(
            network, links, direct_nodes, indirect_nodes)
        sol = optimizer.solve(log_output=True)
        if sol.solve_status.name=='OPTIMAL_SOLUTION':            
            current_objective_value = sol.get_objective_value()
            
            print("Solution status is {}".format(optimizer.solve_details.status))
            print("Objective value for the solution is {}".format(current_objective_value))
            print("Number of variables {}". format(sol.number_of_var_values))
            if min_objective_value > current_objective_value:
                min_objective_value=current_objective_value
                variables_in_sol=sol.as_df()
                    # Write all input to excel
                fname = r'/Model_Stats.xlsx'
                book = wb.create_workbook(BASE_DIR+fname)
                print("Minimum Objective Value obtained {}".format(min_objective_value))
                book = wb.write_link_details(book, links)
                book = wb.write_demand_details(book, demands_dict)
                book = wb.write_solution(book,variables_in_sol)
                wb.save_book(book, BASE_DIR+fname)
            obj_per_epoch[epoch]=current_objective_value
            min_obj_per_epoch[epoch] = min_objective_value
    return obj_per_epoch, min_objective_value, min_obj_per_epoch



M = round(len(network.nodes)*0.7)
""" x1 = []
y1 = []
y2=[]
obj_per_epoch, min_objective_value,min_obj_per_epoch = run_optimiser(network, links,200,M)
print(min_obj_per_epoch)
for obj in min_obj_per_epoch:
    x1.append(obj+1)
    y1.append(min_obj_per_epoch[obj])
    y2.append(obj_per_epoch[obj])


plot.figure(figsize=(9, 4), dpi=80)
color = 'tab:blue'
plot.plot(x1, y1, color=color, label="Global Minimum")
color1 = 'tab:red'
plot.plot(x1, y2, color=color1, label="Local Minimum")
plot.xticks(np.arange(1, 210,20))
    # plot.yscale('log')
plot.yticks(np.arange(0, 350,20))
plot.xlabel("Number of Epochs", fontsize=18)
plot.ylabel('Objective Value', fontsize=16)
plot.legend(loc='upper right', fancybox=True, fontsize=12)
plot.tight_layout()
plot.savefig("Minimumobjective1.png", format='png', pad_inches=0) """
obj_episode={}
min_obj_per_M={}
while(M>=2):
    obj_per_epoch, min_objective_value,min_obj_per_epoch = run_optimiser(network, links,3000,M)
    obj_episode[M] =[obj_per_epoch]
    min_obj_per_M[M]= min_objective_value
    M=M-1
print(min_obj_per_M)







