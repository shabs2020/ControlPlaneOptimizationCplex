import cplex_input
import cplex
from docplex.mp.model import Model
import numpy as np

node_file = "/home/shabnam/PycharmProjects/Doktorarbeit/CPDesignOptimization/example_nodes.json"
edge_file = "/home/shabnam/PycharmProjects/Doktorarbeit/CPDesignOptimization/example_links.json"
network = cplex_input.create_network(node_file, edge_file)


#def model_optimizer(network):

links = {}
i = 0
for e in network.edges:
    links[e] = ['e'+str(i), 1]
    i += 1

##############################################################################

#                   create a CPLEX object

##############################################################################

optimizer = Model(name="minimize cost of control plane network design")
CPLEX_LP_PARAMETERS = {
    'benders.tolerances.feasibilitycut': 1e-9,
    'benders.tolerances.optimalitycut': 1e-9
}

optimizer.parameters.barrier.algorithm = 3
optimizer.parameters.optimalitytarget=3
optimizer.parameters.benders.tolerances.feasibilitycut = CPLEX_LP_PARAMETERS[
    'benders.tolerances.feasibilitycut']
optimizer.parameters.benders.tolerances.optimalitycut = CPLEX_LP_PARAMETERS[
    'benders.tolerances.optimalitycut']

##############################################################################

#                     Initialize input data

##############################################################################

direct_nodes = cplex_input.select_direct_nodes(network=network)
indirect_nodes = list(set(list(network.nodes)).difference(direct_nodes))

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
                                                    ub=10.00, name=varnames_link_capacity)


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

varnames_i_to_j = [
    'u' + i + j for i in indirect_nodes for j in direct_nodes]
var_uij = optimizer.integer_var_dict(lb=0, ub=1,
                                    keys=N_direct_nodes*N_indirect_nodes, name=varnames_i_to_j)
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

##############################################################################

#            Constraint of equation 1: Capacity constraint
#
#                       for all  links e ∈ E:

#           sum(d ∈ D) sum(p ∈ p(d)) δedip*xdip - Ye <= 0
#           Total number of constraints of equation 1: |E|

###############################################################################
quadractic_expr=[]
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
    val = [var_del_di_p[e*N_paths + x] for x in listOfNumbers]

    try:
        if len(ind) == len(val):
            var_sum = np.dot(ind,val)
            
            # optimizer.add(var_sum<=var_link_capacity[e])
            # quadractic_expr.append(var_sum<=var_link_capacity[e])
            # print(quadractic_expr)
         
            # optimizer.add_constraint(ct=var_link_capacity[e]>=0)
           # optimizer.add_constraint(ct=0<=np.prod(val)<=1)
            optimizer.add_constraint(ct=sum(ind[i])-var_link_capacity[e]<=0.0, ctname=constraint_name_1)

        else:
            raise ValueError(
                "Length of variables and coefficients do not match")
    except ValueError as exp:
        print("Error", exp)
#optimizer.add_quadratic_constraints(quadractic_expr)

######################################################################################

#               Constraint of type 2: Flow conservation constraint

#                 forall demands d ∈ D :

#                    sum(p ∈ p(d)) xdp >= h(d)
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
    var_sum = optimizer.sum(ind[i]*val[i] for i in range(len(listOfNumbers)))
    optimizer.add_constraint(ct=var_sum <= float(
        demands_dict[d][1]), ctname=constraint_name_2)

######################################################################################

#               Constraint of type 3: Node disjoint constraint

#                 forall indirect nodes i :

#                    sum(p ∈ p(d)) xdip >= h(di)
#                 Total number of constraints of type 2: |D|

######################################################################################

for i in range(len(indirect_nodes)):
    constraint_name_3 = 'c_u_' + indirect_nodes[i]

    listOfNumbers = [i*len(direct_nodes)+j for j in range(len(direct_nodes))]
    ind = [var_uij[l] for l in listOfNumbers]
    val = [1.0] * len(listOfNumbers)
    var_sum = optimizer.sum(ind[i]*val[i] for i in range(len(listOfNumbers)))
    optimizer.add_constraint(ct=var_sum >= 2.0, ctname=constraint_name_3)

######################################################################################

#               Constraint of type 4: Path diversity constraint

#                 for all demands di :
#                      x_dip-h(d)/sum(u_ij) <=0
#                       sum(u_ij) can be replaced as the number of paths

#                    x_dip<=h(d)/N(P(d_i))
#                 Total number of constraints of type 2: |D|

######################################################################################

for d in demands_dict:

    paths_per_demand = [i[0] for i in demands_dict[d][2]]
    # print(paths_per_demand)
    for p in paths_per_demand:
        constraint_name_4 = 'c4_' + demands_dict[d][0] + p
        newstr = ''.join((ch if ch in '0123456789.-e' else ' ')
                        for ch in p)
        listOfNumbers = [int(i) for i in newstr.split()]
        ind = [var_xdp[p] for p in listOfNumbers]
        val = [1.0] * len(listOfNumbers)
        var_sum = optimizer.sum(ind[i]*val[i]
                                for i in range(len(listOfNumbers)))
        optimizer.add_constraint(ct=var_sum <= float(
            demands_dict[d][1]), ctname=constraint_name_4)

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
for i in indirect_nodes:
    paths = [l for l in demands_dict[i][2]]

    for j in direct_nodes:
        node_exists = [(x, y) for x, y in paths if y == j]
        if len(node_exists) > 0:

            newstr = ''.join((ch if ch in '0123456789' else ' ')
                            for e in demands_dict[i][2][node_exists[0]] for ch in e)
            listOfNumbers = [int(i) for i in newstr.split()]
            sum_link_var=optimizer.sum(var_link_capacity[e] for e in listOfNumbers)
            link_var.append(sum_link_var)

        else:
            link_var.append(0)
        u_ij_var.append(
            var_uij[keys.index(i) * len(direct_nodes) + direct_nodes.index(j)])

optimizer.minimize(np.dot(link_var, [var_uij[i] for i in range(len(link_var))])+optimizer.sum(2*u_ij_var))

    #return optimizer, var_link_capacity, var_xdp, var_uij, var_del_di_p

optimizer.solve()

solution= optimizer.solve_details.status
print(solution)


def run_optimiser():
    optimizer, var_link_capacity, var_xdp, var_uij, var_del_di_p=model_optimizer(network)
    optimizer.solve()
    
def write_solver_status(optimizer, var_link_capacity, var_xdp, var_uij, var_del_di_p):
    solution= optimizer.solution()

