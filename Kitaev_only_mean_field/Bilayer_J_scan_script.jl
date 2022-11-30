using HDF5 
using LinearAlgebra

# sets the real space lattice vectors
a1 = [1/2, sqrt(3)/2]
a2 = [1/2, -sqrt(3)/2]

# This section calculates the dual vectors and makes a matrix of vectors in the Brillouin zone
N = 24
g1, g2 = dual(a1,a2)
BZ = brillouinzone(g1,g2,N,false) # N must be even
half_BZ = brillouinzone(g1,g2,N,true)

# This section sets the run parameters 
K = -1
G = 2
tolerance = 6.0
tol_drop_iteration = 750

Num_scan_points = 400
J_min = -2
J_max = 2

J_perp_list = [0.5]
for J_perp in J_perp_list 
    J_scan_and_save_data(half_BZ,Num_scan_points,J_min,J_max,K,G,J_perp,tolerance,tol_drop_iteration)
    display("completed for J_perp=$J_perp")
end