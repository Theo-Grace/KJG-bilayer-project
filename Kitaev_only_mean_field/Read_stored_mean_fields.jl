using HDF5 
using LinearAlgebra

println("Which parameter scan do you want to read?")
println("Choices: J , G , J_perp")
scan_type = string(readline(stdin))
display(scan_type)
doc_name = "parameter_scan_data/"*scan_type*"_scans"
display(doc_name)

fid = h5open(doc_name,"r")
display(keys(fid))

println("Which set of parameters do you want to read?") 
group_name = readline()

while !(group_name in keys(fid))
    println("Not found, try again.")
    global group_name = readline()
end
g = fid[group_name]

read_fields = read(g["output_mean_fields"])
read_description = read(g["Description_of_run"])
read_coupling_list = read(g[scan_type*"_list"])
read_marked_with_x = read(g["marked_with_x_list"])
close(fid)

title_str = "Varying "*scan_type*" with"*group_name
xlab = scan_type*" /|K|"

corrected_fields , corrected_coupling_list = remove_marked_fields(read_fields,read_coupling_list,read_marked_with_x)
plot_mean_fields_vs_coupling(corrected_fields,corrected_coupling_list,title_str,xlab)
plot_oscillating_fields_vs_coupling(read_fields,read_marked_with_x,read_coupling_list)
