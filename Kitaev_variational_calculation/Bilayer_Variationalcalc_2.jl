# This is the version of the variational calculation for the bilayer Kitaev model as of 26/09/23

# This version uses SVD of the M matrix rather than eigen decomposition of the Hamiltonian
# This optimises the code for the gapless Kitaev model 
# This version will also calculate the correct signs for hopping amplitudes to properly account for interference effects

using LinearAlgebra
using SparseArrays
using Arpack 
using PyPlot
using SkewLinearAlgebra
pygui(true) 

# sets the real space lattice vectors
a1 = [1/2, sqrt(3)/2]
a2 = [-1/2, sqrt(3)/2]

# sets the nearest neighbour vectors 
nz = (a1 + a2)/3
ny = (a1 - 2a2)/3
nx = (a2 - 2a1)/3

nn = [nx,ny,nz] # stores the nearest neighbours in a vector

# sets default boundary conditions
L1 = 50
L2 = 45
m = 1
BCs = [L1,L2,m]

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

function get_M0(BCs)
    """
    Calculates a L1L2 x L1L2 matrix M which is part of the Hamiltonian for a flux free Kitaev model in terms of Majorana fermions 

    uses the lattice vectors 
    a1 = [1/2, sqrt(3)/2]
    a2 = [-1/2, sqrt(3)/2]

    and numbering convention i = 1 + n1 + N*n2 to represent a site [n1,n2] = n1*a1 + n2*a2 

    This assumes periodic boundary conditions with a torus with basis L1*a1, L2*a2 + M*a1
    """
    L1 = BCs[1]
    L2 = BCs[2]
    M = BCs[3]

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

function flip_bond_variable(M,BCs,bond_site,bond_flavour)
    """
    Given part of the Hamiltonian M this returns a new M with a reversed sign for the bond variable at site bond_site with orientation bond_flavour  
    """
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    C_A_index = 1 + bond_site[1] + L1*bond_site[2]

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
    M_flipped=zeros(L1*L2,L1*L2)
    M_flipped[:,:] = M
    M_flipped[C_A_index,C_B_index] = -1
    #M[C_A_index,C_B_index] = 1

    # NOTE: There's a bug here. Using this function seems to change the input M as well as the output for some reason. 
    return M_flipped
end 

function get_M_for_single_visons(BCs)
    """
    Flips a chain of link variables along the boundary
    """

    L1 = BCs[1]
    L2 = BCs[2]
    M = BCs[3]

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

    final_index = Int(round(L1/2))
    A_flipped = A

    for id = 2:(final_index-1)
        A_flipped[id,id] = -1 # This flips the z links on the chain 
        A_flipped[id+1,id] = -1 # This flips the x links 
    end 

    M[1:L1,1:L1] = A_flipped

    display(A_flipped)
    return M 
end 

function get_X_and_Y(BCs)
    M0 = get_M0(BCs)
    M1 = flip_bond_variable(M0,BCs,[1,1],"z") # M1 has z link flipped 
    M0 = get_M0(BCs)
    M2 = flip_bond_variable(M0,BCs,[1,2],"z") # M2 has x link flipped at the same site

    F1 = svd(M1)
    F2 = svd(M2)

    U1 = F1.U
    U2 = F2.U
    V1 = (F1.Vt)'
    V2 = (F2.Vt)'

    U12 = U1'*U2
    V12 = V1'*V2

    X12 = 0.5(U12+V12)
    Y12 = 0.5(U12-V12)

    return X12, Y12
end 

function get_U_and_V(M)
    F = svd(M)
    return F.U ,(F.Vt)'
end 

function calculate_D(BCs,U,V,Num_flipped_bonds)
    """
    This calculates D = prod_i D_i. This term appears in the projection operator. Any state with D eigenvalue of -1 is unphysical. 
    D depends on: 
    - The boundary conditions - 'Theta' term
    - The link configuration 
    - The determinant of the transformation between c majoranas and the fermions diagonalising H 
    """
    theta = BCs[1]+BCs[2] + BCs[3]*(BCs[1]-1)

    exponent = theta + Num_flipped_bonds

    matter_factor = Int(round(det(U)*det(V),digits=1))
    if matter_factor == -1
        exponent +=1
    end 

    if exponent%2 == 0 
        D=1
        display("matter fermion free ground state for $Num_flipped_bonds flipped links is physical")
    else
        D=-1
        display("matter fermion free ground state for $Num_flipped_bonds flipped links is unphysical, must have odd number of matter fermions")
    end

    return D
end 

function get_Heisenberg_hopping(BCs)

    initial_flux_site = [0,0]
    M0 = get_M0(BCs)
    U0, V0 = get_U_and_V(M0)
    calculate_D(BCs,U0,V0,0) 

    M1 = flip_bond_variable(M0,BCs,initial_flux_site,"z") # M1 has z link flipped 
    M0 = get_M0(BCs)
    M2 = flip_bond_variable(M0,BCs,initial_flux_site+[1,0],"z") 

    U1 , V1 = get_U_and_V(M1)
    U2 , V2 = get_U_and_V(M2)

    calculate_D(BCs,U1,V1,1)
    calculate_D(BCs,U2,V2,1)

    U12 = U1'*U2
    V12 = V1'*V2

    X12 = 0.5(U12+V12)
    Y12 = 0.5(U12-V12)

    M12 = inv(X12)*Y12

    U21 = U2'*U1
    V21 = V2'*V1

    X21 = 0.5(U21+V21)
    Y21 = 0.5(U21-V21)

    M21 = inv(X21)*Y21

    X1=0.5*(U1'+V1')
    Y1=0.5*(U1'-V1')
    T1= [X1 Y1 ; Y1 X1]
    M1 = inv(X1)*Y1

    X2 = 0.5*(U2'+V2')
    Y2 = 0.5*(U2'-V2')
    T2 = [X2 Y2 ;Y2 X2]
    T = T2*T1'
    M2 = inv(X2)*Y2

    X = T[1:Int(size(T)[1]/2),1:Int(size(T)[1]/2)]
    Y = T[1:Int(size(T)[1]/2),(Int(size(T)[1]/2)+1):end]
    M = inv(X)*Y

    initial_C_A_index = 1 + initial_flux_site[1] + L1*initial_flux_site[2]

    hop = abs(det(X12))^(0.5)*(((U1*M21*V1')-(U1*V1'))[initial_C_A_index+1,initial_C_A_index]+1)
    
    display(abs(det(X12))^(0.5))
    #display((U12*V12'-(U12*V12')'))
    display(Pfaffian(V12*U12'-(V12*U12')'))

    U10 = U1'*U0
    V10 = V1'*V0
    U20 = U2'*U0
    V20 = V2'*V0

    X10 = 0.5*(U10+V10)
    Y10 = 0.5*(U10-V10)
    X20 = 0.5*(U20+V20)
    Y20 = 0.5*(U20-V20)

    M10 = inv(X10)*Y10
    M20 = inv(X20)*Y20
    #display(det(X10)*(det(I-M20*M10)^(0.5)))
    display(abs(det(X10))*(Pfaffian([0.5*(M20-M20') -I ; I -0.5*(M10-M10') ])))

    #display(pfaffian([0.5*(M20-M20') -I ; I -0.5*(M10-M10') ]))
    #display(det([0.5*(M20-M20') -I ; I -0.5*(M10-M10') ])^(0.5))
    
    return hop
end 

function hop_single_vison(BCs,M_initial,initial_site)
    M1 = flip_bond_variable(M_initial,BCs,initial_site,"z")
    M_final = flip_bond_variable(M1,BCs,initial_site+[1,0],"x")
    return M_final
end 

function plot_GS_energy_vs_vison_seperation(BCs)
    M0 = get_M0(BCs)
    M_initial = flip_bond_variable(M0,BCs,[0,0],"z")

    E_fluxless = -sum(svd(M0).S)/(BCs[1]*BCs[2])

    Energy_vs_seperation = zeros(L1)
    Energy_vs_seperation[1] = -sum(svd(M_initial).S)
    M = M_initial

    for n1 = 1:(L1-1)
        M = hop_single_vison(BCs,M,n1*[1,0])
        Energy_vs_seperation[n1+1] = -sum(svd(M).S)
    end 

    Energy_vs_seperation = Energy_vs_seperation./(BCs[1]*BCs[2])

    plot((1:L1)./L1,Energy_vs_seperation)
    plot((1:L1)./L1,E_fluxless*ones(L1))
end 

