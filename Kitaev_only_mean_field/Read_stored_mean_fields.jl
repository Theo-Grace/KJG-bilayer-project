using HDF5 
using LinearAlgebra

function read_stored_data(scan_type)
    """
    Takes scan_type as an argument which can only be "J" "G" or "J_perp"
    """
    doc_name = "parameter_scan_data/"*scan_type*"_scans"
    display(doc_name)
    
    fid = h5open(doc_name,"r")
    show(stdout,"text/plain",keys(fid))

    read_another=true

    while read_another == true
        println("Which file do you want to read? Copy the file name. Type N to quit") 
        group_name = readline()

        if group_name == "N"
            break
        end

        while !(group_name in keys(fid))
            println("Not found, try again.")
            group_name = readline()
        end

        g = fid[group_name]

        read_fields = read(g["output_mean_fields"])
        read_description = read(g["Description_of_run"])
        read_coupling_list = read(g[scan_type*"_list"])
        read_marked_with_x = read(g["marked_with_x_list"])

        title_str = "Varying "*scan_type*" with"*group_name
        xlab = scan_type*" /|K|"

        corrected_fields , corrected_coupling_list = remove_marked_fields(read_fields,read_coupling_list,read_marked_with_x)
        plot_mean_fields_vs_coupling(corrected_fields,corrected_coupling_list,title_str,xlab)
        plot_oscillating_fields_vs_coupling(read_fields,read_marked_with_x,read_coupling_list)
    end 
    close(fid)

    return read_fields , read_description , read_coupling_list , read_marked_with_x 
end