# This is the version of the variational calculation for the bilayer Kitaev model as of 26/09/23

using LinearAlgebra
using SparseArrays
using Arpack 
using PyPlot
pygui(true) 

# sets the real space lattice vectors
a1 = [1/2, sqrt(3)/2]
a2 = [-1/2, sqrt(3)/2]

# sets the nearest neighbour vectors 
nz = (a1 + a2)/3
ny = (a1 - 2a2)/3
nx = (a2 - 2a1)/3

nn = [nx,ny,nz] # stores the nearest neighbours in a vector

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

function get_M0(L1,L2,M)
    """
    Calculates a L1L2 x L1L2 matrix M which is part of the Hamiltonian for a flux free Kitaev model in terms of Majorana fermions 

    uses the lattice vectors 
    a1 = [1/2, sqrt(3)/2]
    a2 = [-1/2, sqrt(3)/2]

    and numbering convention i = 1 + n1 + N*n2 to represent a site [n1,n2] = n1*a1 + n2*a2 

    This assumes periodic boundary conditions with a torus with basis L1*a1, L2*a2 + M*a1
    """
    N = L1*L2
    A = zeros(L1,L1)
    B = zeros(L1,L1)
    for j = 1:L1-1
        A[j,j] = 1
        A[j+1,j] = 1
        B[j,j] = 1
    end 
    A[L1,L1] = 1
    A[1,L1] = 1
    B[L1,L1] = 1
    B_prime = zeros(L1,L1)
    B_prime[:,1:M] = B[:,(L1-M+1):L1]
    B_prime[:,(M+1):L1] = B[:,1:(L1-M)]

    M = zeros(N,N)
    for j = 1:(L2-1)
        M[(1+(j-1)*L1):(j*L1),(1+(j-1)*L1):(j*L1)] = A
        M[(1+j*L1):((j+1)*L1),(1+(j-1)*L1):(j*L1)] = B
    end

    M[L1*(L2-1)+1:N,L1*(L2-1)+1:N] = A
    M[1:L1,(L1*(L2-1)+1):N] = B_prime
    return M 
end 

function flip_bond_variable(M,L1,L2,m,bond_site,bond_flavour)
    """
    Given a Hamiltonian H this returns a new Hamiltonian with a reversed sign for the bond variable at site bond_site with orientation bond_flavour  
    """

    C_A_index = 1 + bond_site[1] + N*bond_site[2]

    if bond_flavour == "z"
        C_B_index = C_A_index
    elseif bond_flavour == "y"
        if bond_site[2] == 0
            C_B_index = L1*(L2-1) + m + C_A_index
        else
            C_B_index = C_A_index - L1
        end 
    else
        if bond_site[1] == 0 
            C_B_index = C_A_index + L1 -1
        else 
            C_B_index = C_A_index -1 
        end 
    end 

    M[C_A_index,C_B_index] = - M[C_A_index,C_B_index]

    return M
end 

L1 = 20
L2 = 20 
m = 1