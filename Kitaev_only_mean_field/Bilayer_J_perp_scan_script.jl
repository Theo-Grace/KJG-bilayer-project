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
J = 0
#G = 0.50865298455 # interesting points at 0.50285 0.5086525298455
tolerance = 7.0
tol_drop_iteration = 750

Num_scan_points = 100
J_perp_min = 0.3
J_perp_max = 0.7

for G in [0.1,0.2,0.3]
    J_perp_scan_and_save_data(half_BZ,Num_scan_points,J_perp_min,J_perp_max,K,J,G,tolerance,tol_drop_iteration)
end

#=
fid = h5open("parameter_scan_data/test_file","r+")
create_group(fid,"test_group")
g = fid["test_group"]
write(g,"test_dataset",stored_fields)
write(g,"test_description",Description)
write(g,"test_parameter_list",J_perp_list)
close(g)
close(fid)

fid = h5open("parameter_scan_data/test_file","r")
dset = fid["test_group/test_dataset"]
read_fields = read(dset)
dset = fid["test_group/test_description"]
read_description = read(dset)
dset = fid["test_group/test_parameter_list"]
read_J_perp_list = read(dset)
close(fid)
=#