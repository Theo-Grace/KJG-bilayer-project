#=
This is a specialised form of the code to calculate the mean fields for a Kitaev bilayer model focused on stacking arrangements
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
K = 1

# defines "spin matrices" used throughout the calculation
Mx = [0 1 0 0 ;-1 0 0 0 ;0 0 0 -1; 0 0 1 0]
My = [0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0]
Mz = [0 0 0 1; 0 0 -1 0; 0 1 0 0 ; -1 0 0 0]
M_alpha = [Mx,My,Mz]

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
Large_BZ = brillouinzone(g1,g2,100,false)

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

function Kitaev_block(Ux_mean_fields,k)
    """
    creates a 4x4 matrix mathcal{M}^K which is a block in the Hamiltonian corresponing to the Kitaev interaction
    """
    MK = zeros(Complex{Float64},4,4)
    for alpha = [1,2,3]
        MK += -im*exp(-im*dot(k,nn[alpha]))*M_alpha[alpha]*transform_bond_type(Ux_mean_fields,alpha)*M_alpha[alpha]
    end 
    return MK
end

function AB_interlayer_block(U_interlayer_mean_fields)
    M_perp = zeros(Complex{Float64},4,4)
    for alpha = [1,2,3]
        M_perp += -im*M_alpha[alpha]*U_interlayer_mean_fields*M_alpha[alpha]
    end 
    return M_perp
end 

function AB_Hamiltonian(Ux_layer_1,Ux_layer_2,U_interlayer,k)
    MK1 = Kitaev_block(Ux_layer_1,k)
    MK2 = Kitaev_block(Ux_layer_2,k)
    M_perp = AB_interlayer_block(U_interlayer)

    H_AB = [0 MK1 0 M_perp ; MK1' 0 0 0 ; 0 0 0 transpose(MK2) ; M_perp' 0 conj(MK2) 0]
    return H_AB 
end 

function update_mean_fields(Ux_layer_1,Ux_layer_2,U_interlayer)
    new_Ux_1 = zeros(Complex{Float64},0,0)
    new_Ux_2 = zeros(Complex{Float64},0,0)
    new_U_interlayer = zeros(Complex{Float64},0,0)
    for k in BZ
        H_AB = AB_Hamiltonian(Ux_layer_1,Ux_layer_2,U_interlayer,k)
        R, O = diagonalise(H_AB)
        all_fields = R*O*R'
        new_Ux_1 += exp(-im*dot(k,nn[1]))*all_fields[1:4,5:8]
        new_Ux_2 += exp(-im*dot(k,nn[1]))*all_fields[9:12,13:16]
        new_U_interlayer += all_fields[1:4,13:16]
    end 

    return im*new_Ux_1/N, im*new_Ux_2/N , im*new_U_interlayer/N
end 

