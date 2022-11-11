#=
This is a specialised form of the code to calculate the mean fields for a KJG bilayer model
ASSUMPTIONS:
- No magnetic ordering
- Isotropic couplings (all bond types equivalent)

=#

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

mean_fields = [-1 0 0 0 ; 0 1 0 0 ; 0 0 0.5 0.25 ; 0 0 0.25 0.5] + 0.0001*rand(4,4)

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
    Qx = diagm([1,1,1,1])
    Qy = [1 0 0 0 ; 0 0 1 0 ; 0 1 0 0 ; 0 0 0 1]
    Qz = [1 0 0 0 ; 0 0 0 1 ; 0 0 1 0 ; 0 1 0 0]
    Q = [Qx,Qy,Qz]

    return Q[alpha]*mean_fields*Q[alpha]
end

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

function Hamiltonian_combined(mean_fields,k,nn,K=-1,J=0,G=0)
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

mean_fields_interlayer_i = [1 0 0 0 ; 0 -0.25 0  0 ; 0 0 -0.25 0 ; 0 0 0 -0.25]
mean_fields_interlayer_j = [1 0 0 0 ; 0 -0.25 0  0 ; 0 0 -0.25 0 ; 0 0 0 -0.25]
mean_fields_intralayer_1 = [-1 0 0 0 ; 0 1 0 0 ; 0 0 0.5 0.25 ; 0 0 0.25 0.5]
mean_fields_intralayer_2 = [-1 0 0 0 ; 0 1 0 0 ; 0 0 0.5 0.25 ; 0 0 0.25 0.5]

Mean_fields = [ mean_fields_intralayer_1 mean_fields_interlayer_i ; mean_fields_interlayer_j mean_fields_intralayer_2 ] + 0.1*rand(8,8)

function Hamiltonian_interlayer(Mean_fields,J_perp)
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
        updated_mean_fields[:,:] -= (Fourier_bilayer(U*occupancymatrix*U',-k,nn[1]))
    
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

function run_bilayer_to_convergence(half_BZ,initial_mean_fields,nn,tolerance=10.0,K=-1,J=0,G=0,J_perp=0)
    """
    Given a set of 16x16x3 mean fields and half BZ this repeatedly updates mean fields by calculating the Hamiltonian from initial mean fields and returning the mean fields they generate. 
    Checks for convergence by calculating the difference in real part of the mean field matrix between two iterations, and checking that there is no element for which the difference is larger than the specified tolerance. 
    """
    old_mean_fields = initial_mean_fields
    new_mean_fields = zeros(8,8)
    old_old_mean_fields = zeros(8,8)
    tol = 10^(-tolerance)
    it_num = 0 
    not_converged = true
    not_oscillating = true 
    while not_converged
        new_mean_fields = update_bilayer_mean_fields(half_BZ,old_mean_fields,nn,K,J,G,J_perp)
        diff= abs.(new_mean_fields-old_mean_fields)
        diff2 = abs.(new_mean_fields - old_old_mean_fields)
        not_converged = any(diff .> tol)
        println(it_num)
        not_oscillating = any(diff2 .> tol)
        if not_oscillating == false
            println("Oscillating solution")
            break
        end
        it_num +=1 
        old_old_mean_fields = old_mean_fields
        old_mean_fields = new_mean_fields
    end
    return round.(new_mean_fields,digits=trunc(Int,tolerance))
end

function get_bilayer_bandstructure(BZ,mean_fields,nn,K=[1,1,1],J=0,G=0,J_perp=0)
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