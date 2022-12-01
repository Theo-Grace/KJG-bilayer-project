#=
This is a specialised form of the code to calculate the mean fields for a KJG bilayer model
ASSUMPTIONS:
- No magnetic ordering
- Isotropic couplings (all bond types equivalent)

=#
using HDF5
using LinearAlgebra
using PyPlot # This is the library used for plotting 
pygui(true) # This changes the plot backend from julia default to allow plots to be displayed

# sets the real space lattice vectors
a1 = [1/2, sqrt(3)/2]
a2 = [1/2, -sqrt(3)/2]

# sets the nearest neighbour vectors 
nx = (a1 - a2)/3
ny = (a1 + 2a2)/3
nz = -(2a1 + a2)/3

nn = [nx,ny,nz] # stores the nearest neighbours in a vector 

# sets the coupling parameters
K = -[1,1,1]
J = 0
G = 0

function antisymmetrise(A)
    """
    Takes a square matrix A and returns a matrix with the lower triangle set to minus the upper triangle of A 
    """
    dim = convert(Int,sqrt(length(A)))
    for i in 2:dim
        for j in 1:i-1
            A[i,j]=-A[j,i]
        end
    end
    return A 
end

function dual(A1,A2)
    """
    Calculates the 2D dual vectors for a given set of 2D lattice vectors 
    """
    U = [A1[1] A1[2]; A2[1] A2[2]] 
    V = 2*pi*inv(U)

    v1 = [V[1] V[2]]
    v2 = [V[3] V[4]]

    return v1, v2 
end

function brillouinzone(g1, g2, N, half=true)
    
    """
    Generate a list of N x N k-vectors in the brillouin zone, with an option to do a half-BZ.
    N must be EVEN.
    Vectors are chosen to avoid BZ edges and corners to simplify integrating quantities over the BZ.
    """

    M = floor(Int64, N/2)

    dx, dy = g1/N, g2/N
    upper = M - 1
    
    if half == true
        lowerx = 0
        lowery = - M
    else
        lowerx = - M
        lowery = - M
    end

    return [(ix+0.5)*dx + (iy+0.5)*dy for ix in lowerx:upper, iy in lowery:upper]
end

# This section calculates the dual vectors and makes a matrix of vectors in the Brillouin zone
N = 24
g1, g2 = dual(a1,a2)
BZ = brillouinzone(g1,g2,N,false) # N must be even
half_BZ = brillouinzone(g1,g2,N,true)

mean_fields = [-1 0 0 0 ; 0 1 0 0 ; 0 0 0.5 0.25 ; 0 0 0.25 0.5] + 0.01*rand(4,4)

function Fourier(M,k,neighbour_vector)
    """
    Given an 8x8 matrix M this returns the matrix with off diagonal 4x4 blocks multiplied by phase factors e^ik.n
    This implements a Fourier transform from mean fields in real space to momentum space
    """
    phase = exp(im*dot(k,neighbour_vector))
    F = Diagonal([1,1,1,1,phase,phase,phase,phase])
    return (F')*M*F
end

function transform_bond_type(mean_fields,alpha)
    """
    Given a 4x4 matrix of mean fields <iX_iX_j^T> calculated with a bond ij of type x, this transforms the mean fields to the equivalent matrix if it had been calculated with bond type alpha = x, y, z
    """
    Qx = diagm([1,1,1,1])
    Qy = [1 0 0 0 ; 0 0 1 0 ; 0 1 0 0 ; 0 0 0 1]
    Qz = [1 0 0 0 ; 0 0 0 1 ; 0 0 1 0 ; 0 1 0 0]
    Q = [Qx,Qy,Qz]

    return Q[alpha]*mean_fields*Q[alpha]
end

# This set of functions calculates seperate terms in the Hamiltonian assuming isotropic coupling constants
#=
function get_M(mean_fields,alpha)
    Tx = [0 -1 0 0 ; 1 0 0 0 ; 0 0 0 0 ; 0 0 0 0]
    Ty = [0 0 -1 0 ; 0 0 0 0; 1 0 0 0; 0 0 0 0]
    Tz = [0 0 0 -1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 0]
    T = [Tx,Ty,Tz]
    
    return T[alpha]*mean_fields*T[alpha]
end

function Hamiltonian_K(mean_fields,k,nn,K=-1)
    M = zeros(8,8)
    H_K = zeros(Complex{Float64},8,8)
    for alpha = 1:3
        M[1:4,5:8] = get_M(transform_bond_type(mean_fields,alpha),alpha)
        M[5:8,1:4] = -transpose(M[1:4,5:8])
        H_K += 0.5im*K*Fourier(M,k,nn[alpha])
    end
    return H_K
end

function Hamiltonian_J(mean_fields,k,nn,J=1)
    H_J = zeros(Complex{Float64},8,8)
    for alpha = 1:3
        M = zeros(8,8)
        for beta = 1:3
            M[1:4,5:8] += get_M(transform_bond_type(mean_fields,alpha),beta)
            M[5:8,1:4] = -transpose(M[1:4,5:8])
        end
        H_J += 0.5im*J*Fourier(M,k,nn[alpha])
    end
    return H_J
end

function get_M_G(mean_fields,alpha)
    Tx = [0 -1 0 0 ; 1 0 0 0 ; 0 0 0 0 ; 0 0 0 0]
    Ty = [0 0 -1 0 ; 0 0 0 0; 1 0 0 0; 0 0 0 0]
    Tz = [0 0 0 -1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 0]
    T = [Tx,Ty,Tz]
    
    beta = setdiff([1,2,3],alpha)
    return T[beta[1]]*mean_fields*T[beta[2]] +T[beta[2]]*mean_fields*T[beta[1]] 
end

function Hamiltonian_G(mean_fields,k,nn,G=1)
    M_G = zeros(8,8)
    H_G = zeros(Complex{Float64},8,8)
    for alpha = 1:3
        M_G[1:4,5:8] = get_M_G(transform_bond_type(mean_fields,alpha),alpha)
        M_G[5:8,1:4] = -transpose(M_G[1:4,5:8])
        H_G += 0.5im*G*Fourier(M_G,k,nn[alpha])
    end
    return H_G
end

function Hamiltonian_full(mean_fields,k,nn,K=-1,J=0,G=0)
    """
    Calculates the full monolayer Hamiltonian by adding seperate terms
    """
    H = Hamiltonian_K(mean_fields,k,nn,K)
    if J != 0 
        H += Hamiltonian_J(mean_fields,k,nn,J)
    end
    if G != 0 
        H += Hamiltonian_G(mean_fields,k,nn,G)
    end 
    return H
end    
=# 

function Hamiltonian_combined(mean_fields,k,nn,K=-1,J=0,G=0)
    """
    Takes:
    - a 4x4 matrix of mean fields <iX_iX_j^T> calculated with a bond ij of type x
    - 2D wavevector k 
    - nearest neighbour vectors nn
    - isotropic coupling constants K J G
    Returns
    - an 8x8 Hamiltonian including all 3 terms
    """

    # These matricies can be used to tranform the original mean field matrix into the form that it appears in the Hamiltonian
    Tx = [0 -1 0 0 ; 1 0 0 0 ; 0 0 0 0 ; 0 0 0 0]
    Ty = [0 0 -1 0 ; 0 0 0 0; 1 0 0 0; 0 0 0 0]
    Tz = [0 0 0 -1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 0]
    T = [Tx,Ty,Tz]
    
    H = zeros(Complex{Float64},8,8)
    for alpha = 1:3

        M_tot = zeros(4,4)

        mean_fields_alpha_bond = transform_bond_type(mean_fields,alpha)
        
        M_tot += (J+K)T[alpha]*mean_fields_alpha_bond*T[alpha]

        for beta = setdiff([1,2,3],alpha)
            gamma = setdiff([1,2,3],[alpha,beta])[1]

            M_tot += J*T[beta]*mean_fields_alpha_bond*T[beta]
            M_tot += G*T[beta]*mean_fields_alpha_bond*T[gamma]
        end

        H[1:4,5:8] += 0.5*im*M_tot*exp(im*dot(k,nn[alpha]))
    end
    H[5:8,1:4] = H[1:4,5:8]'
    return H 
end

function fermi_dirac(E,temp=0.0,mu=0)
    """
    Given a scalar energy E it returns the probability for a fermionic state of that energy to be occupied
    """
    return (exp((E-mu)/temp) + 1)^-1
end

function diagonalise(H,temp=0.0,mu=0)
    """
    Diagonalises the Hamiltonian H
    returns:
    - a unitary matrix with eigenvectors as columns U
    - The occupancy matrix O giving the probability for the state to be occupied 
    """
    U = eigvecs(H)
    E = eigvals(H)
    O =  Diagonal(fermi_dirac.(E,temp,mu))
    return U , O 
end

function update_mean_fields(half_BZ,old_mean_fields,nn,K=-1,J=0,G=0)
    """
    calculates an updated mean field matrix. This calculates the Hamiltonian from a given set of mean fields,
    then diagonalises the Hamiltonian and calculates a set of mean fields from that Hamiltonian
    Requires:
    - Brillouin zone as a matrix of k vectors
    - a current mean field 4x4 matrix 
    - nearest neighbour vectors nn 
    - Coupling parameters K, J, G (default is isotropic K and J=G=0)
    returns
    - new mean fields as a real 4x4 matrix
    """
    Num_unit_cells = 2*length(half_BZ)
    updated_mean_fields = zeros(Complex{Float64},8,8)
    for k in half_BZ
        H = Hamiltonian_combined(old_mean_fields,k,nn,K,J,G)
        U , occupancymatrix = diagonalise(H)

        updated_mean_fields[:,:] += Fourier(transpose(U')*occupancymatrix*transpose(U),k,nn[1])
        updated_mean_fields[:,:] -= Fourier(U*occupancymatrix*U',-k,nn[1])
    end
    updated_mean_fields = (im.*updated_mean_fields)./Num_unit_cells
 
    return real.(updated_mean_fields[1:4,5:8])
end

function run_to_convergence(half_BZ,initial_mean_fields,nn,tolerance=10.0,K=-1,J=0,G=0)
    old_mean_fields = initial_mean_fields
    new_mean_fields = zeros(8,8,3)
    tol = 10^(-tolerance)
    it_num = 0 
    not_converged = true
    while not_converged
        new_mean_fields = update_mean_fields(half_BZ,old_mean_fields,nn,K,J,G)
        diff= abs.(new_mean_fields-old_mean_fields)
        not_converged = any(diff .> tol)
        it_num +=1 
        old_mean_fields = new_mean_fields
        println(it_num)
    end
    return new_mean_fields
end

# This section calculates the bandstructure given a set of mean fields by calculating the eigenvalues of the corresponding Hamiltonian for each wavevector.
# These functions can be used to plot interesting directions in the brillouin zone 
function get_bandstructure(BZ,mean_fields,nn,K=-1,J=0,G=0)
    """
    takes:
    - The BZ as a matrix of k vectors
    - an 4x4 matrix of mean fields mean_fields 
    - nearest neighbour vectors in a vector nn
    - Kitaev coupling parameters as a scalar K
    - Heisenberg coupling J as a scalar
    - Gamma coupling G as a scalar 
    returns:
    - A dictionary called bandstructure with a key for each k in the Brillouin zone whose entries are a vector
    containing the energies for that k 
    """
    bandstructure = Dict()
    for k in BZ 
        H = Hamiltonian_full(mean_fields,k,nn,K,J,G)
        bandstructure[k] = eigvals(H)
    end
    return bandstructure
end

function plot_bands_G_to_K(BZ,bandstructure,bilayer=false)
    """
    This plots the bands along the direction between Gamma and M points in the Brillouin Zone
    This requires:
    - the bandstructure as a dictionary bandstructure[k] whose entries are the 8 energies for that k vector
    - the Brillouin zone BZ as a matrix of k vectors 
    - a boolean argument bilayer which determines whether there are 16 or 8 bands to plot 
    """
    if bilayer == false
        num_majoranas = 8
    else
        num_majoranas = 16
    end

    GtoK = []
    bands_GtoK = [[] for i = 1:num_majoranas]
    for k in BZ 
        if (k[2] == 0) 
            push!(GtoK,k)
            for i in 1:num_majoranas
                push!(bands_GtoK[i],bandstructure[k][i])
            end
        end 
    end 
    kGtoK = (1:length(GtoK))*(g1[1]+g2[1])/(2*length(GtoK))
    for i in 1:num_majoranas
        plot(kGtoK,bands_GtoK[i],color="black")
    end
    title( "Between \$\\Gamma\$ and \$ K \$ points")
    ylabel("Energy")
    xlabel("Wavevector")

    #This section sets x,y axis limits to make easier comparison to results from "Dynamics of QSL beyond integrability, J. Knolle et al"
    ax = gca()
    ax[:set_xlim]([3.14,5.3])
    ax[:set_ylim]([0,1])
end

function plot_bands_G_to_M(BZ,bandstructure)
    """
    This plots the bands along the direction between Gamma and K points in the BZ 
    This requires:
    - the bandstructure as a dictionary bandstructure[k] whose entries are the 8 energies at that k value
    - the Brillouin zone BZ as a matrix of k vectors 
    """
    GtoM =[]
    bands_GtoM = [[] for i = 1:8]
    for k in BZ 
        if (k[1]==0)
            push!(GtoM,k)
            for i in 1:8
                push!(bands_GtoM[i],bandstructure[k][i])
            end
        end 
    end 
    for i in 1:8
        plot(1:length(GtoM),bands_GtoM[i])
    end
    title("Majorana bandstructure between \$\\Gamma\$ and \$ M \$ points")
    ylabel("Energy")
    xlabel("Wavevector")
    #legend(["matter","gauge","gauge","gauge","gauge","gauge","gauge","matter"])
end

function converge_and_plot(BZ,half_BZ,initial_mean_fields,nn,tolerance=10.0,K=-1,J=0,G=0)
    """
    Combines functions to allow you to calculate final fields and plot the bands at once 
    returns the converged mean fields 
    """
    final_mean_fields = run_to_convergence(half_BZ,initial_mean_fields,nn,tolerance,K,J,G)
    bandstructure = get_bandstructure(BZ,final_mean_fields,nn,K,J,G)
    plot_bands_G_to_K(BZ,bandstructure)
    return final_mean_fields
end

# This section adds functions to treat the bilayer model 
# Mean fields in this section are given by 8x8 matrices. The intralayer mean fields are 4x4 on diagonal blocks and the off diagonal blocks are interlayer mean fields 

mean_fields_interlayer_i = [1 0 0 0 ; 0 0.25 0  0 ; 0 0 0.25 0 ; 0 0 0 0.25] # <i X_1i X_2i^T>
mean_fields_interlayer_j = [-1 0 0 0 ; 0 -0.25 0  0 ; 0 0 -0.25 0 ; 0 0 0 -0.25] # <i X_1j X_2j^T>
mean_fields_intralayer_1 = [-1 0 0 0 ; 0 1 0 0 ; 0 0 0.5 0.25 ; 0 0 0.25 0.5] # <i X_1i X_1j^T>
mean_fields_intralayer_2 = [-1 0 0 0 ; 0 1 0 0 ; 0 0 0.5 0.25 ; 0 0 0.25 0.5] # <i X_2i X_2j^T>

Mean_fields = [ mean_fields_intralayer_1 mean_fields_interlayer_i ; mean_fields_interlayer_j mean_fields_intralayer_2 ] + 0.05*rand(8,8)

function Hamiltonian_interlayer(Mean_fields,J_perp)
    """
    Creates the 16x16 interlayer Hamiltonian.
    ASSUMPTIONS:
    - AA stacking
    - Heisenberg interactions between nearest neighbours only
    - isotropic monolayer interactions 
    - no spin ordering 
    """
    H_perp = zeros(Complex{Float64},16,16)

    Tx = [0 -1 0 0 ; 1 0 0 0 ; 0 0 0 0 ; 0 0 0 0]
    Ty = [0 0 -1 0 ; 0 0 0 0; 1 0 0 0; 0 0 0 0]
    Tz = [0 0 0 -1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 0]
    T = [Tx,Ty,Tz]

    for alpha = 1:3
        H_perp[1:4,9:12] += 0.5*im*J_perp*(T[alpha]*Mean_fields[1:4,5:8]*T[alpha])
        H_perp[5:8,13:16] += 0.5*im*J_perp*(T[alpha]*Mean_fields[5:8,1:4]*T[alpha])
    end

    H_perp[9:16,1:8] = H_perp[1:8,9:16]'

    return H_perp
end

function Hamiltonian_intralayer(Mean_fields,k,nn,K=-1,J=0,G=0)
    """
    Produces a 16x16 Hamiltonian containing the Hamiltonian for a KJG monolayer in the 2 8x8 on diagonal blocks
    ASSUMPTIONS:
    - isotropic monolayer interactions 
    - no spin ordering 
    """
    H_intra = zeros(Complex{Float64},16,16)

    H_intra[1:8,1:8] = Hamiltonian_combined(Mean_fields[1:4,1:4],k,nn,K,J,G)
    H_intra[9:16,9:16] = Hamiltonian_combined(Mean_fields[5:8,5:8],k,nn,K,J,G)

    return H_intra
end

function update_bilayer_mean_fields(half_BZ,old_mean_fields,nn,K=-1,J=0,G=0,J_perp=0)
    """
    calculates an updated mean field matrix. This calculates the Hamiltonian from a given set of mean fields,
    then diagonalises the Hamiltonian and calculates a set of mean fields from that Hamiltonian
    Requires:
    - Brillouin zone as a matrix of k vectors
    - a current mean field 16x16x3 matrix 
    - nearest neighbour vectors nn 
    - Coupling parameters K, J, G, J_perp (default is isotropic K and J=G=0)
    returns
    - a new mean field 16x16x3 matrix
    """
    Num_unit_cells = 2*length(half_BZ)
    updated_mean_fields = zeros(Complex{Float64},16,16)
    H_inter = Hamiltonian_interlayer(old_mean_fields,J_perp)

    for k in half_BZ
        H = Hamiltonian_intralayer(old_mean_fields,k,nn,K,J,G) + H_inter
        U , occupancymatrix = diagonalise(H)
    
        updated_mean_fields[:,:] += Fourier_bilayer(transpose(U')*occupancymatrix*transpose(U),k,nn[1])
        updated_mean_fields[:,:] += conj(Fourier_bilayer(U*occupancymatrix*U',-k,nn[1])) # I changed this from a - sign to a + conj 
    
    end

    updated_mean_fields = (im.*updated_mean_fields)./Num_unit_cells

    return real.([ updated_mean_fields[1:4,5:8] updated_mean_fields[1:4,9:12] ; updated_mean_fields[5:8,13:16] updated_mean_fields[9:12,13:16] ])
end

function Fourier_bilayer(M,k,neighbour_vector)
    """
    Given an 16x16 matrix M 
    This implements a Fourier transform from mean fields in real space to momentum space
    """
    phase = exp(im*dot(k,neighbour_vector))
    F = Diagonal([1,1,1,1,phase,phase,phase,phase,1,1,1,1,phase,phase,phase,phase])
    return (F')*M*F
end

function run_bilayer_to_convergence(half_BZ,initial_mean_fields,nn,tolerance=10.0,K=-1,J=0,G=0,J_perp=0,tol_drop_iteration=500)
    """
    Given a set of 16x16x3 mean fields and half BZ this repeatedly updates mean fields by calculating the Hamiltonian from initial mean fields and returning the mean fields they generate. 
    Checks for convergence by calculating the difference in real part of the mean field matrix between two iterations, and checking that there is no element for which the difference is larger than the specified tolerance. 
    Checks for oscillating solutions by calculating the second difference. This must be less than 100x smaller than the specified tolerance for 10 successive iterations to be identified as oscillatory
    """
    old_mean_fields = initial_mean_fields
    new_mean_fields = zeros(8,8)
    old_old_mean_fields = zeros(8,8)
    tol = 10^(-tolerance)
    it_num = 0 
    osc_num = 0
    not_converged = true
    not_oscillating = true 
    mark_with_x = false
    while not_converged
        new_mean_fields = update_bilayer_mean_fields(half_BZ,old_mean_fields,nn,K,J,G,J_perp)
        diff= abs.(new_mean_fields-old_mean_fields)
        diff2 = abs.(new_mean_fields - old_old_mean_fields)
        not_converged = any(diff .> tol)
        println(it_num)

        not_oscillating = any(diff2 .> 0.01*tol)
        if not_oscillating == false
            osc_num += 1
            println("osc number is $osc_num")
            display(diff.*(diff .>tol))
            display(diff2.*(diff2 .>tol))
        else
            osc_num = 0
        end

        if osc_num >= 10
            println("Oscillating solution")
            mark_with_x = true
            break
        end
        if it_num%tol_drop_iteration == 0 && it_num >0
            tol = 10*tol
            println(tol)
            mark_with_x = true
        end

        it_num +=1 
        old_old_mean_fields = old_mean_fields
        old_mean_fields = new_mean_fields
    end
    return round.(new_mean_fields,digits=trunc(Int,tolerance)) , mark_with_x
end

function get_bilayer_bandstructure(BZ,mean_fields,nn,K=-1,J=0,G=0,J_perp=0)
    """
    takes:
    - The BZ as a matrix of k vectors
    - an 16x16x3 matrix of mean fields mean_fields 
    - nearest neighbour vectors in a vector nn
    - Kitaev coupling parameters as a vector K
    - Heisenberg coupling J as a scalar
    - Gamma coupling G as a scalar 
    returns:
    - A dictionary called bandstructure with a key for each k in the Brillouin zone whose entries are a vector
    containing the energies for that k 
    """
    bandstructure = Dict()
    H_inter = Hamiltonian_interlayer(mean_fields,J_perp)
    for k in BZ 
        H = Hamiltonian_intralayer(mean_fields,k,nn,K,J,G) + H_inter
        bandstructure[k] = eigvals(H)
    end
    return bandstructure
end

function bilayer_converge_and_plot(BZ,half_BZ,initial_Mean_fields,nn,tolerance=10.0,K=-1,J=0,G=0,J_perp=0)
    """
    combines functions to take initial 16x16x3 mean fields and iterate until they converge, and then plots the band structure. 
    """
    final_mean_fields = run_bilayer_to_convergence(half_BZ,initial_Mean_fields,nn,tolerance,K,J,G,J_perp)
    bandstructure = get_bilayer_bandstructure(BZ,final_mean_fields,nn,K,J,G,J_perp)
    plot_bands_G_to_K(BZ,bandstructure,true)
    return final_mean_fields
end


# This section contains functions to do parameter scans 
function scan_J_coupling(stored_mean_fields, half_BZ, nn,J_list,tolerance = 10.0, K=-1,G=0,J_perp=0,tol_drop_iteration=500)
    num_J_values = size(J_list)[1]
    marked_with_x = [false for _ in 1:num_J_values]
    title_str = "Varying J with couplings K=$K \$\\Gamma\$=$G \$J_{\\perp}\$=$J_perp"
    xlab="J/|K|"

    for (id,J) in enumerate(J_list)
        stored_mean_fields[:,:,id] = fix_signs(stored_mean_fields[:,:,id],K,J,G,J_perp)
        stored_mean_fields[:,:,id] , mark_with_x = run_bilayer_to_convergence(half_BZ,stored_mean_fields[:,:,id],nn,tolerance,K,J,G,J_perp,tol_drop_iteration)
        marked_with_x[id] = mark_with_x
        #plot_mean_fields_vs_coupling(stored_mean_fields[:,:,1:id],J_list[1:id],title_str,xlab)
        #plot_mean_fields_check(stored_mean_fields[:,:,id],J_list[id],mark_with_x,K,G,J_perp,0,false)
        display(id)
    end
    return stored_mean_fields , marked_with_x
end

function scan_G_coupling(stored_mean_fields, half_BZ, nn,G_list,tolerance = 10.0, K=-1,J=0,J_perp=0,tol_drop_iteration=500)
    num_G_values = size(G_list)[1]
    marked_with_x = [false for _ in 1:num_G_values]
    title_str = "Varying \$\\Gamma\$ with couplings K=$K J=$J \$J_{\\perp}\$=$J_perp"
    xlab="\$\\Gamma\$/|K|"

    for (id,G) in enumerate(G_list)
        stored_mean_fields[:,:,id] = fix_signs(stored_mean_fields[:,:,id],K,J,G,J_perp)
        stored_mean_fields[:,:,id] , mark_with_x = run_bilayer_to_convergence(half_BZ,stored_mean_fields[:,:,id],nn,tolerance,K,J,G,J_perp,tol_drop_iteration)
        marked_with_x[id] = mark_with_x
        if id%25 ==0 
            plot_mean_fields_vs_coupling(stored_mean_fields[:,:,1:id],G_list[1:id],title_str,xlab)
        end
        #plot_mean_fields_check(stored_mean_fields[:,:,id],G_list[id],mark_with_x,K,J,J_perp,0,false)
        display(id)
    end

    return stored_mean_fields , marked_with_x
end

function scan_J_perp_coupling(stored_mean_fields, half_BZ, nn,J_perp_list,tolerance = 10.0, K=-1,J=0,G=0,tol_drop_iteration=500)
    """
    Takes:
    - a matrix of 8x8xN random fields, with the signs fixed according to the chosen convention. 
    - a list of N J_perp values at which the mean fields will be calculated
    Returns:
    - an 8x8xN matrix of mean fields calculated to the specified tolerance
    - a list of boolean values marked_with_x. A true value in this list indicates that the solution was either oscillatory or calculated with higher tolerance  
    """
    num_J_perp_values = size(J_perp_list)[1]
    marked_with_x = [false for _ in 1:num_J_perp_values]
    title_str = "Varying \$J_{\\perp}\$ with couplings K=$K J=$J \$\\Gamma\$=$G"
    xlab="\$J_{\\perp}\$/|K|"

    for (id,J_perp) in enumerate(J_perp_list)
        stored_mean_fields[:,:,id] = fix_signs(stored_mean_fields[:,:,id],K,J,G,J_perp)
        stored_mean_fields[:,:,id] , mark_with_x = run_bilayer_to_convergence(half_BZ,stored_mean_fields[:,:,id],nn,tolerance,K,J,G,J_perp,tol_drop_iteration)
        marked_with_x[id] = mark_with_x
        plot_mean_fields_vs_coupling(stored_mean_fields[:,:,1:id],J_perp_list[1:id],title_str,xlab)
        plot_mean_fields_check(stored_mean_fields[:,:,id],J_perp_list[id],mark_with_x,K,J,G,0,false)
        display(id)
    end
    
    return stored_mean_fields , marked_with_x
end

function plot_mean_fields_vs_coupling(stored_mean_fields,coupling_list,title_str="",xlab="")

    plot(coupling_list,stored_mean_fields[1,1,:],color="blue") # uC calculated on x bond
    plot(coupling_list,stored_mean_fields[2,2,:],color="purple") # uK calculated on x bond
    plot(coupling_list,stored_mean_fields[3,3,:],color="red") # uJ calculated on x bond
    plot(coupling_list,stored_mean_fields[3,4,:],color="orange") # uG calculated on x bond 
    plot(coupling_list,stored_mean_fields[2,6,:],color="green") # interlayer mean field mixing x gauge bonds on the same side 

    title(title_str)
    xlabel(xlab)
    ylabel("mean fields")
    legend(["\$u_{ij}^0\$","\$u_{ij}^x\$","\$u_{ij}^y\$","\$ u_{ij}^{yz}\$","\$ u_{12}^x\$"])
end

function plot_oscillating_fields_vs_coupling(stored_mean_fields,marked_with_x,coupling_list)
    """
    Plots only the points that were marked x during calculation, indicating either oscillatory solution or that it was calculated at higher tolerance than otherwise specified. 
    Takes:
    - A complete list of 8x8xN calculated mean fields
    - A list marked_with_x which specifies which points to plot 
    - A list of couplings 
    NOTE: This function should be used to overlay the results of plot_mean_fields_vs_coupling to show which points were oscillatory
    """

    for (id,mark_with_x) in enumerate(marked_with_x)
        if mark_with_x == true
            scatter(coupling_list[id],stored_mean_fields[1,1,id],marker = "x",color="blue")
            scatter(coupling_list[id],stored_mean_fields[2,2,id],marker = "x",color="purple")
            scatter(coupling_list[id],stored_mean_fields[3,3,id],marker = "x",color="red")
            scatter(coupling_list[id],stored_mean_fields[3,4,id],marker = "x",color="orange")
            scatter(coupling_list[id],stored_mean_fields[2,6,id],marker = "x",color="green")
        end
    end 
end

# This section adds functions to save the scan data to a file 
function J_perp_scan_and_save_data(half_BZ,Num_scan_points,J_perp_min,J_perp_max,K,J,G,tolerance,tol_drop_iteration=500)
    
    # sets the nearest neighbour vectors 
    nx = (a1 - a2)/3
    ny = (a1 + 2a2)/3
    nz = -(2a1 + a2)/3

    nn = [nx,ny,nz] # stores the nearest neighbours in a vector 

    J_perp_list = collect(LinRange(J_perp_min,J_perp_max,Num_scan_points))

    N=sqrt(2*length(half_BZ))

    Description = "The parameters used in this run were K=$K J=$J G=$G. 
    The Brillouin zone resolution parameter was N=$N.
    The tolerance used when checking convergence was tol=10^(-$tolerance).
    The number of J_perp values used was $Num_scan_points between $J_perp_min and $J_perp_max.
    Each point was calculated from random initial fields.
    The signs of random fields were fixed.
    Points marked x were either oscillating solutions or ran for over $tol_drop_iteration iterations and were calculated with higher tolerance.
    The marked_with_x_list is true for values of J_perp which were marked"

    initial_mean_fields = 0.5*rand(8,8,Num_scan_points)
    stored_mean_fields , marked_with_x = scan_J_perp_coupling(initial_mean_fields, half_BZ, nn,J_perp_list,tolerance, K,J,G,tol_drop_iteration)

    group_name = "K=$K"*"_J=$J"*"_G=$G"*"_J_perp=[$J_perp_min,$J_perp_max]_data"

    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\KJG MF data\\parameter_scan_data\\J_perp_scans","cw")
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
end

function G_scan_and_save_data(half_BZ,Num_scan_points,G_min,G_max,K,J,J_perp,tolerance,tol_drop_iteration=500)
    
    # sets the nearest neighbour vectors 
    nx = (a1 - a2)/3
    ny = (a1 + 2a2)/3
    nz = -(2a1 + a2)/3

    nn = [nx,ny,nz] # stores the nearest neighbours in a vector 

    G_list = collect(LinRange(G_min,G_max,Num_scan_points))

    N=sqrt(2*length(half_BZ))

    Description = "The parameters used in this run were K=$K J=$J J_perp=$J_perp. 
    The Brillouin zone resolution parameter was N=$N.
    The tolerance used when checking convergence was tol=10^(-$tolerance).
    The number of J_perp values used was $Num_scan_points between $G_min and $G_max.
    Each point was calculated from random initial fields.
    The signs of random fields were fixed.
    Points marked x were either oscillating solutions or ran for over $tol_drop_iteration iterations and were calculated with higher tolerance.
    The marked_with_x_list is true for values of J_perp which were marked"

    initial_mean_fields = 0.5*rand(8,8,Num_scan_points)
    stored_mean_fields , marked_with_x = scan_G_coupling(initial_mean_fields, half_BZ, nn,G_list,tolerance, K,J,J_perp,tol_drop_iteration)

    group_name = "K=$K"*"_J=$J"*"_J_perp=$J_perp"*"_G=[$G_min,$G_max]_data"

    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\KJG MF data\\parameter_scan_data\\G_scans","cw")
    it_num = 2
    while group_name in keys(fid)
        global group_name = group_name*"_$it_num"
    end

    create_group(fid,group_name)
    g = fid[group_name]
    write(g,"output_mean_fields",stored_mean_fields)
    write(g,"Description_of_run",Description)
    write(g,"G_list",G_list)
    write(g,"marked_with_x_list",marked_with_x)
    close(fid)
end

function J_scan_and_save_data(half_BZ,Num_scan_points,J_min,J_max,K,G,J_perp,tolerance,tol_drop_iteration=500)
    
    # sets the nearest neighbour vectors 
    nx = (a1 - a2)/3
    ny = (a1 + 2a2)/3
    nz = -(2a1 + a2)/3

    nn = [nx,ny,nz] # stores the nearest neighbours in a vector 

    J_list = collect(LinRange(J_min,J_max,Num_scan_points))

    N=sqrt(2*length(half_BZ))

    Description = "The parameters used in this run were K=$K G=$G J_perp=$J_perp. 
    The Brillouin zone resolution parameter was N=$N.
    The tolerance used when checking convergence was tol=10^(-$tolerance).
    The number of J_perp values used was $Num_scan_points between $J_min and $J_max.
    Each point was calculated from random initial fields.
    The signs of random fields were fixed.
    Points marked x were either oscillating solutions or ran for over $tol_drop_iteration iterations and were calculated with higher tolerance.
    The marked_with_x_list is true for values of J_perp which were marked"

    initial_mean_fields = 0.5*rand(8,8,Num_scan_points)
    stored_mean_fields , marked_with_x = scan_J_coupling(initial_mean_fields, half_BZ, nn,J_list,tolerance, K,G,J_perp,tol_drop_iteration)

    group_name = "K=$K"*"_G=$G"*"_J_perp=$J_perp"*"_J=[$J_min,$J_max]_data"

    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\KJG MF data\\parameter_scan_data\\J_scans","cw")
    it_num = 2
    while group_name in keys(fid)
        global group_name = group_name*"_$it_num"
    end

    create_group(fid,group_name)
    g = fid[group_name]
    write(g,"output_mean_fields",stored_mean_fields)
    write(g,"Description_of_run",Description)
    write(g,"J_list",J_list)
    write(g,"marked_with_x_list",marked_with_x)
    close(fid)
end

function remove_marked_fields(stored_fields,coupling_list,marked_with_x)
    marked_list = []
    N = length(coupling_list)
    for (id,mark_with_x) = enumerate(marked_with_x)
        if mark_with_x == true
            append!(marked_list , id)
        end
    end

    correct_ids = setdiff(1:N,marked_list)

    corrected_stored_fields = stored_fields[:,:,correct_ids]
    corrected_coupling_list = coupling_list[correct_ids]
    return corrected_stored_fields , corrected_coupling_list
end



# This section adds functions to check the code is converging correctly.
function check_random_inputs(num_checks,half_BZ,tolerance,K,J,G,J_perp)
    """
    This generates a set of N=num_checks random 8x8 mean field matrices and then then iterates each one to convergence and plots the resulting mean fields as a scatter plot. This allows you to check the distribution of outputs,
    and check that the code is converging consistently. 
    """
    stored_fields = 0.5*rand(8,8,num_checks)
    
    #=
    # This fixes the signs to be consistent with the Kitaev exact solution
    stored_fields[1,1,:] = abs.(stored_fields[1,1,:]) 
    stored_fields[5,5,:] = abs.(stored_fields[5,5,:])
    stored_fields[2,2,:] = -abs.(stored_fields[2,2,:])
    stored_fields[6,6,:] = -abs.(stored_fields[6,6,:])

    # This chooses a sign convention to avoid oscillating solutions involving interlayer terms 
    stored_fields[1,5,:] = abs.(stored_fields[1,5,:])
    stored_fields[2,6,:] = abs.(stored_fields[2,6,:])
    stored_fields[3,7,:] = abs.(stored_fields[3,7,:])
    stored_fields[4,8,:] = abs.(stored_fields[4,8,:])

    stored_fields[5,1,:] = -abs.(stored_fields[5,1,:])
    stored_fields[6,2,:] = -abs.(stored_fields[6,2,:])
    stored_fields[7,3,:] = -abs.(stored_fields[7,3,:])
    stored_fields[8,4,:] = -abs.(stored_fields[8,4,:])

    # This chooses consist signs to avoid oscillating solutions with Heisenberg term 
    stored_fields[3,3,:] = abs.(stored_fields[3,3,:])
    stored_fields[4,4,:] = abs.(stored_fields[4,4,:])
    stored_fields[7,7,:] = abs.(stored_fields[7,7,:])
    stored_fields[8,8,:] = abs.(stored_fields[8,8,:])
    =#

    stored_fields = fix_signs(stored_fields,K,J,G,J_perp)

    check_list = 1:num_checks
    for N in check_list
        stored_fields[:,:,N] , mark_with_x = run_bilayer_to_convergence(half_BZ,stored_fields[:,:,N],nn,tolerance,K,J,G,J_perp)
        plot_mean_fields_check(stored_fields[:,:,N],check_list[N],mark_with_x,K,J,G,J_perp)
        display(stored_fields[:,:,N])
    end

    return stored_fields
end

function fix_signs(random_fields,K,J,G,J_perp)
    """
    Takes an 8x8xN array of matrices and fixes the signs of certain terms to avoid oscillating solutions in which the fields do not converge to a fixed set but instead flip between fields with differing sign. 
    NOTE: This makes a choice of sign convention 
    """
    num_matrices = size(random_fields,3)

    sign_fixing_mask = ones(8,8,num_matrices)
    sign_fixing_mask[2,2,:] .= -1
    sign_fixing_mask[6,6,:] .= -1
    for j = 1:4
        sign_fixing_mask[4+j,j,:] .= -1
    end
   
    if G < 0 # Found that for G<0 need to have uG the same sign as uK for converging solution
        sign_fixing_mask[3,4,:] .=-1
        sign_fixing_mask[4,3,:] .=-1
        sign_fixing_mask[7,8,:] .=-1
        sign_fixing_mask[8,7,:] .=-1
    end 

    if J < 0 # Found that negative J need to have uJ same sign as uK to converge 
        for j = 1:2
            sign_fixing_mask[2+j,2+j,:] .= -1
            sign_fixing_mask[6+j,6+j,:] .= -1
        end
    end

    if J>1
        sign_fixing_mask[2,2,:] .= 1
        sign_fixing_mask[6,6,:] .= 1
    end

    #This section fixes the signs for the KJG model with J>0 G>0 J_perp>0 
    for beta = [1,2,3]
        for gamma = setdiff([1,2,3],beta)
            sign_fixing_mask[1+beta,5+gamma,:].=-1
        end
    end

    for j = 1:2
        sign_fixing_mask[2,2+j,:].=-1
        sign_fixing_mask[2+j,2,:].=-1
        sign_fixing_mask[6,6+j,:].=-1
        sign_fixing_mask[6+j,6,:].=-1
    end

    # This section suppresses the solutions involving mixed gauge and matter sector, which give oscillatory solutions for J>0.
    if J> 0.3
        for j = [1,3]
            sign_fixing_mask[1,1+j,:].=-1
            sign_fixing_mask[5,5+j,:].=-1
        end  
        sign_fixing_mask[3,1,:].=-1  
        sign_fixing_mask[7,5,:].=-1
    end

    return sign_fixing_mask.*random_fields
end

function plot_mean_fields_check(stored_mean_fields,check_list,mark_with_x,K,J,G,J_perp,ttl_str=true)
    """
    This plots mean fields calculated for a fixed set of coupling constants using a set of 8x8xN random initial matricies against N.
    If all initial conditions converge to the same solution then this plot should be a series of horizontal lines.
    """
    if mark_with_x == true
        mark = "x" # oscillating solutions and solutions at lower tolerance are marked by a x 
    elseif ttl_str == true
        mark = "o"  
    elseif ttl_str ==false
        mark = " "
    end

    scatter(check_list,stored_mean_fields[1,1,:],marker = mark,color="blue")
    scatter(check_list,stored_mean_fields[2,2,:],marker = mark,color="purple")
    scatter(check_list,stored_mean_fields[3,3,:],marker = mark,color="red")
    scatter(check_list,stored_mean_fields[3,4,:],marker = mark,color="orange")
    scatter(check_list,stored_mean_fields[2,6,:],marker = mark,color="green")

    if ttl_str == true
        title("Checking convergence for parameters K=$K J=$J \$\\Gamma\$=$G \$J_{\\perp}\$=$J_perp")
        xlabel("Check number")
        ylabel("mean fields")
        legend(["\$u_{ij}^0\$","\$u_{ij}^x\$","\$u_{ij}^y\$","\$ u_{ij}^{yz}\$","\$ u_{12}^x\$"])
    end
end

# This section adds functions to read stored data from a HDF5 filefunction read_stored_data(scan_type)
function read_stored_data(scan_type)
    """
    Reads data stored in a HDF5 file.
    Each type of parameter scan (J,G,J_perp) has it's own HDF5 file. 
    Takes:
    - scan_type as an argument which can only be "J" "G" or "J_perp" which must be entered as a string 
    Returns:
    - The mean fields from the final read. 
    - Plots the parameter scan *corrected* fields meaning that any x marked points are removed. The x marks are plotted seperately. 
    """
    doc_name = homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\KJG MF data\\parameter_scan_data\\"*scan_type*"_scans"
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

        title_str = "Varying "*scan_type*" with "*group_name[1:4]*" "*group_name[6:8]*" "*group_name[10:end-5]
        xlab = scan_type*" /|K|"

        corrected_fields , corrected_coupling_list = remove_marked_fields(read_fields,read_coupling_list,read_marked_with_x)
        plot_mean_fields_vs_coupling(corrected_fields,corrected_coupling_list,title_str,xlab)
        #plot_oscillating_fields_vs_coupling(read_fields,read_marked_with_x,read_coupling_list)
    end 
    close(fid)

    return read_fields , read_description , read_coupling_list , read_marked_with_x 
end