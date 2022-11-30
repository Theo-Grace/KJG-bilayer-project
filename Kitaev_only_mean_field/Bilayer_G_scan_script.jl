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
J = 0.3
tolerance = 6.0
tol_drop_iteration = 750

Num_scan_points = 400
G_min = -2
G_max = 2

# Interesting transition occurs at J_perp = 0.11750 0.1167 for G~0.50285 and J_perp = 0.01923 for G~0.50865
# The transition depends sensitively on G. For G~0.5086298455 it occurs at J_perp ~0.008-9


J_perp_list = [0]
for J_perp in J_perp_list 
    G_scan_and_save_data(half_BZ,Num_scan_points,G_min,G_max,K,J,J_perp,tolerance,tol_drop_iteration)
    display("completed for J_perp=$J_perp")
end