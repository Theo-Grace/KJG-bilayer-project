using HDF5 
using LinearAlgebra

# sets the real space lattice vectors
a1 = [1/2, sqrt(3)/2]
a2 = [1/2, -sqrt(3)/2]

# sets the nearest neighbour vectors 
nx = (a1 - a2)/3
ny = (a1 + 2a2)/3
nz = -(2a1 + a2)/3

nn = [nx,ny,nz] # stores the nearest neighbours in a vector 

# This section calculates the dual vectors and makes a matrix of vectors in the Brillouin zone
N = 24
g1, g2 = dual(a1,a2)
BZ = brillouinzone(g1,g2,N,false) # N must be even
half_BZ = brillouinzone(g1,g2,N,true)

# This section sets the run parameters 
K = -1
J = 0
G = 0
tolerance = 6.0
tol_drop_iteration = 500

Num_points = 50
J_perp_min = 0.44
J_perp_max = 0.6
J_perp_list = collect(LinRange(J_perp_min,J_perp_max,Num_points)) 

Description = "The parameters used in this run were K=$K J=$J G=$G. 
The Brillouin zone resolution parameter was N=$N.
The tolerance used when checking convergence was tol=10^(-$tolerance).
The number of J_perp values used was $Num_points between $J_perp_min and $J_perp_max.
The signs of random fields were fixed.
Points marked x were either oscillating solutions or ran for over $tol_drop_iteration iterations and were calculated with higher tolerance.
The marked_with_x_list is true for values of J_perp which were marked"

initial_mean_fields = fix_signs(0.5*rand(8,8,Num_points))
stored_mean_fields , marked_with_x = scan_J_perp_coupling(initial_mean_fields, half_BZ, nn,J_perp_list,tolerance, K,J,G,tol_drop_iteration)

group_name = "K=$K"*"_J=$J"*"_G=$G"*"_data"

fid = h5open("parameter_scan_data/J_perp_scans","cw")
it_num = 2
while group_name in keys(fid)
    global group_name = group_name*"_$it_num"
end

create_group(fid,group_name)
g = fid[group_name]
write(g,"output_mean_fields",stored_mean_fields)
write(g,"Description_of_run",Description)
write(g,"J_perp_list",J_perp_list)
write(g,"marked_with_x_list",marked_with_x)
close(fid)

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