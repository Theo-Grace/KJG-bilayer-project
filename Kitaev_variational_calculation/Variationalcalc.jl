using LinearAlgebra
using SparseArrays
using Arpack 
using PyPlot
pygui(true) 


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


#=
This section uses the old ordering of the sites, where the periodic boundary conditions are implemented using the basis (n1,n2)
function get_H0(N,K=1)
    """
    Calculates a 2N^2 x 2N^2 matrix H0 which is the Hamiltonian for a flux free Kitaev model in terms of complex matter fermions. 
    - K is the Kitaev parameter (take to be +1 or -1)
    returns:
    - a 2N^2 x 2N^2 Hamiltonian matrix which is real and symmetric 
    """
    A = spzeros(N,N)
    B = spzeros(N,N)

    for j = 1:N-1
        A[j,j] = K
        A[j+1,j] = K
        B[j,j] = K
    end 
    A[N,N] = K
    A[1,N] = K

    

    M = spzeros(N^2,N^2)
    for j = 1:(N-1)
        M[(1+(j-1)*N):(j*N),(1+(j-1)*N):(j*N)] = A
        M[(1+(j-1)*N):(j*N),(1+j*N):((j+1)*N)] = B
    end

    M[N*(N-1)+1:N^2,N*(N-1)+1:N^2] = A
    M[N*(N-1)+1:N^2,1:N] = B

    H = spzeros(2*N^2,2*N^2)

    H[1:N^2,1:N^2] = M + transpose(M)
    H[(N^2+1):2*N^2,1:N^2] = M - transpose(M)
    H[1:N^2,(N^2+1):2*N^2] = transpose(M) - M 
    H[(N^2+1):2*N^2,(N^2+1):2*N^2] = -M - transpose(M) 

    return H
end

function get_HF(N,K=1,flux_site=[Int(round(N/2)),Int(round(N/2))],flux_flavour="z")
    """
    Creates a 2N^2 x 2N^2 Hamiltonian matrix for a flux sector with a single flux pair. 
    The flux pair is located at the lattice site flux_site which is given as a coordinate [n1,n2] = n1*a1 + n2*a2 where a1,a2 are plvs. 
    The lattice sites are numbered such that an A site at [n1,n2] is given the index 1 + n1 + N*n2 
    The B lattice sites are numbered such that the B site connected to A via a z bond is given the same number. 
    flux_flavour specifies the type of bond which connects the fluxes in the pair. It is given as a string, either "x","y" or "z".

    The matrix H is calculated using the following steps:
    - Calculate the matrix M which descibes H in terms of Majorana fermions:
        0.5[C_A C_B][ 0 M ; -M 0][C_A C_B]^T
        
        - M is initially calculated in the same way as for H0 
        - A flipped bond variable (equivalent to adding flux pair) is added
        - flux_site specifies the A sublattice site, which specifies the row of M on which the variable is flipped, via C_A_index = 1 + n1 + n2*N
        - flux_flavour specifies the neighbouring B sublattice site, which specifies the column of M on which the variable is flipped 
    
    - Use M to calculate H in the basis of complex matter fermions:
        C_A = f + f'
        C_B = i(f'-f)

    returns:
    - A 2N^2 x 2N^2 sparse matrix Hamiltonian in the complex matter fermion basis 

    """

    flux_site = [flux_site[1],flux_site[2]]

    
    A = spzeros(N,N)
    B = spzeros(N,N)

    for j = 1:N-1
        A[j,j] = K
        A[j+1,j] = K
        B[j,j] = K
    end 
    A[N,N] = K
    A[1,N] = K
    B[N,N] = K

    M = spzeros(N^2,N^2)
    for j = 1:(N-1)
        M[(1+(j-1)*N):(j*N),(1+(j-1)*N):(j*N)] = A
        M[(1+(j-1)*N):(j*N),(1+j*N):((j+1)*N)] = B
    end

    M[N*(N-1)+1:N^2,N*(N-1)+1:N^2] = A
    M[N*(N-1)+1:N^2,1:N] = B

    C_A_index = 1 + flux_site[1] + N*flux_site[2]

    if flux_flavour == "z"
        C_B_index = C_A_index
    elseif flux_flavour == "y"
        if flux_site[2] == N - 1
            C_B_index = 1 + flux_site[1]
        else
            C_B_index = C_A_index + N 
        end 
    else
        if flux_site[1] == 0 
            C_B_index = C_A_index + N -1
        else 
            C_B_index = C_A_index -1 
        end 
    end 

    M[C_A_index,C_B_index] = -K 

    H = zeros(2*N^2,2*N^2)

    H[1:N^2,1:N^2] = M + transpose(M)
    H[(N^2+1):2*N^2,1:N^2] = M - transpose(M)
    H[1:N^2,(N^2+1):2*N^2] = transpose(M) - M 
    H[(N^2+1):2*N^2,(N^2+1):2*N^2] = -M - transpose(M) 

    return H 

end 

function flip_bond_variable(H,bond_site,bond_flavour)
    N = Int(sqrt(size(H)[1]/2))

    M = get_M_from_H(H)

    C_A_index = 1 + bond_site[1] + N*bond_site[2]

    if bond_flavour == "z"
        C_B_index = C_A_index
    elseif bond_flavour == "y"
        if bond_site[2] == N - 1
            C_B_index = 1 + bond_site[1]
        else
            C_B_index = C_A_index + N 
        end 
    else
        if bond_site[1] == 0 
            C_B_index = C_A_index + N -1
        else 
            C_B_index = C_A_index -1 
        end 
    end 

    M[C_A_index,C_B_index] = - M[C_A_index,C_B_index]

    H = zeros(2*N^2,2*N^2)

    H[1:N^2,1:N^2] = M + transpose(M)
    H[(N^2+1):2*N^2,1:N^2] = M - transpose(M)
    H[1:N^2,(N^2+1):2*N^2] = transpose(M) - M 
    H[(N^2+1):2*N^2,(N^2+1):2*N^2] = -M - transpose(M) 

    return H 
end 

function convert_lattice_vector_to_index(lattice_vector,N)
    index = 1 + lattice_vector[1] + N*lattice_vector[2]
    return index
end 

function find_flipped_bond_coordinates(H)
    """
    Assumes the Hamiltonian contains only a single flipped bond. 
    Finds the coordinate of the flipped bond site in the form [n1,n2] = n1*a1 + n2*a2
    """

    N = Int(sqrt(size(H)[1]/2))
    h = H[1:N^2,1:N^2]
    Delta = H[1:N^2,(1+N^2):2*N^2]

    M = 0.5*(h-Delta)

    K = sign(sum(M))

    bond_indicies_set = findall(x->x==-K,M)
    
    for bond_indicies in bond_indicies_set
        C_A_index = bond_indicies[1]
        if C_A_index % N == 0 
            C_A_n1 = Int(N-1)
            C_A_n2 = Int(C_A_index/N -1)
        else
            C_A_n2 = Int(floor(C_A_index/N))
            C_A_n1 = Int(C_A_index - C_A_n2*N -1)
        end 

        C_B_index = bond_indicies[2]
        if C_B_index % N == 0 
            C_B_n1 = Int(N-1)
            C_B_n2 = Int(C_B_index/N -1) 
        else
            C_B_n2 = Int(floor(C_B_index/N))
            C_B_n1 = Int(C_B_index - C_B_n2*N -1) 
        end 

        if C_A_index == C_B_index
            bond_flavour = "z"
        elseif C_B_n1 == C_A_n1 
            bond_flavour = "y"
        elseif C_B_n2 == C_A_n2
            bond_flavour = "x"
        end
        
        println("flipped bond at R = $C_A_n1 a1 + $C_A_n2 a2 with flavour $bond_flavour")
    end 
end 
=#

function get_H0(N,K=1)
    """
    Calculates a 2N^2 x 2N^2 matrix H0 which is the Hamiltonian for a flux free Kitaev model in terms of complex matter fermions. 
    - K is the Kitaev parameter (take to be +1 or -1)
    returns:
    - a 2N^2 x 2N^2 Hamiltonian matrix which is real and symmetric 
    """
    A = spzeros(N,N)
    B = spzeros(N,N)

    for j = 1:N-1
        A[j,j] = K
        A[j+1,j] = K
        B[j+1,j] = K
    end 
    A[N,N] = K
    A[1,N] = K
    B[1,N] = K

    M = spzeros(N^2,N^2)
    for j = 1:(N-1)
        M[(1+(j-1)*N):(j*N),(1+(j-1)*N):(j*N)] = A
        M[(1+(j-1)*N):(j*N),(1+j*N):((j+1)*N)] = B
    end

    M[N*(N-1)+1:N^2,N*(N-1)+1:N^2] = A
    M[N*(N-1)+1:N^2,1:N] = B

    H = spzeros(2*N^2,2*N^2)

    H[1:N^2,1:N^2] = M + transpose(M)
    H[(N^2+1):2*N^2,1:N^2] = M - transpose(M)
    H[1:N^2,(N^2+1):2*N^2] = transpose(M) - M 
    H[(N^2+1):2*N^2,(N^2+1):2*N^2] = -M - transpose(M) 

    return H
end

function diagonalise_sp(H)
    """
    Diagonalises a given Hamiltonian H. H is assumed to be real, symmetric and in sparse matrix format. 
    returns:
    - a full spectrum of eigenvalues. It is assumed that for each energy E, -E is also an eigenvalue.
    - A unitary (orthogonal as T is real) matrix T which diagonalises H
    """
    N_sq = Int(size(H)[1]/2)
    half_spectrum , eigvecs = eigs(H, nev = N_sq, which =:LR, tol = 10^(-7), maxiter = 1000)

    if det(H) <= 10^(-6) 
        println("Warning: H is singular so there are degenerate eigenvalues")
        println("This may lead to numerical errors")
        display(det(H))
    end 

    energies = zeros(2*N_sq,1)
    energies[1:N_sq] = half_spectrum
    energies[N_sq+1:end] = - half_spectrum

    X = eigvecs[1:N_sq,1:N_sq]'
    Y = eigvecs[(N_sq+1):2*N_sq,1:N_sq]'

    T = [X Y ; Y X]

    plot(energies)

    return energies , T 
end 

function diagonalise(H)
    N_sq = Int(size(H)[1]/2)

    H = Matrix(H)

    energies = eigvals(H)
    reverse!(energies)
    eigvec = eigvecs(H)

    gamma = [zeros(N_sq,N_sq) Matrix(I,N_sq,N_sq) ; Matrix(I,N_sq,N_sq) zeros(N_sq,N_sq)]
    eigvec = gamma*eigvec

    X = eigvec[1:N_sq,1:N_sq]'
    Y = eigvec[(N_sq+1):2*N_sq,1:N_sq]'

    T = [X Y ; Y X]

    return energies , T 
end 

function get_HF(N,K=1,flux_site=[Int(round(N/2)),Int(round(N/2))],flux_flavour="z")
    """
    Creates a 2N^2 x 2N^2 Hamiltonian matrix for a flux sector with a single flux pair. 
    The flux pair is located at the lattice site flux_site which is given as a coordinate [n1,n2] = n1*a1 + n2*a2 where a1,a2 are plvs. 
    The lattice sites are numbered such that an A site at [n1,n2] is given the index 1 + n1 + N*n2 
    The B lattice sites are numbered such that the B site connected to A via a z bond is given the same number. 
    flux_flavour specifies the type of bond which connects the fluxes in the pair. It is given as a string, either "x","y" or "z".

    The matrix H is calculated using the following steps:
    - Calculate the matrix M which descibes H in terms of Majorana fermions:
        0.5[C_A C_B][ 0 M ; -M 0][C_A C_B]^T
        
        - M is initially calculated in the same way as for H0 
        - A flipped bond variable (equivalent to adding flux pair) is added
        - flux_site specifies the A sublattice site, which specifies the row of M on which the variable is flipped, via C_A_index = 1 + n1 + n2*N
        - flux_flavour specifies the neighbouring B sublattice site, which specifies the column of M on which the variable is flipped 
    
    - Use M to calculate H in the basis of complex matter fermions:
        C_A = f + f'
        C_B = i(f'-f)

    returns:
    - A 2N^2 x 2N^2 sparse matrix Hamiltonian in the complex matter fermion basis 

    """

    H0 = get_H0(N,K)

    HF = flip_bond_variable(H0,flux_site,flux_flavour)

    return HF

end 

function get_X_and_Y(T)
    """
    Given a unitary matrix T in the form T = [X* Y* ; Y X] this function extracts the X and Y matrices 
    """
    N_sq = Int(size(T)[1]/2)
    X = T[(N_sq+1):2*N_sq,(N_sq+1):2*N_sq]
    Y = T[(N_sq+1):2*N_sq,1:N_sq]

    return X , Y 
end 

function get_F_matrix(TF1,TF2)
    """
    calculates the matrix F (see Knolle's thesis) used to relate quasi-particle vacua in two different flux sectors 
    """
    T = TF2*TF1'
    X , Y = get_X_and_Y(T)

    F = conj(inv(X)*Y)

    return F 
end 

function get_normalisation_constant(TF1,TF2)
    T = TF2*TF1'
    X , Y = get_X_and_Y(T)

    c = (det(X'*X))^(1/4)

    return c
end 

function calculate_z_Heisenberg_hopping_matrix_element(N,K)
    #=
    HF1 = get_HF(N,K,initial_flux_site)
    EF1, TF1 = diagonalise(HF1)
    HF2 = get_HF(N,K,initial_flux_site+[1,0])
    EF2, TF2 = diagonalise(HF2)

    F = get_F_matrix(TF1,TF2)
    C = get_normalisation_constant(TF1,TF2)

    X1 , Y1 = get_X_and_Y(TF1)

    C_A_index = 1 + initial_flux_site[1]+N*initial_flux_site[2] + 1 # Final +1 is due to the final site being +a1 with respect to the initial site 
    C_B_index = 1 + initial_flux_site[1] + N*initial_flux_site[2]

    a = (X1+Y1)[:,C_A_index]
    b = (Y1-X1)[:,C_B_index]

    println("Hopping due to y spin interaction:")
    display((a'*b - b'*F*a))
    println("Normalisation constant C:")
    display(C)

    hopping_amp = (a'*b - b'*F*a + C)
    =#

    mark_with_x = false
    initial_flux_site = [Int(round(N/2)),Int(round(N/2))]
    final_flux_site = initial_flux_site + [1,0]
    
    initial_HF = get_HF(N,K,initial_flux_site,"z")
    final_HF = get_HF(N,K,final_flux_site,"z")

    _ , initial_TF = diagonalise(initial_HF)
    _ , final_TF = diagonalise(final_HF)

    C = get_normalisation_constant(initial_TF,final_TF)

    initial_X , initial_Y = get_X_and_Y(initial_TF)

    a_init = transpose(initial_X + initial_Y)
    b_init = transpose(initial_X - initial_Y)


    # T is a matrix which transforms the creation and annihilation operators for the initial flux sector into those of the final flux sector [f_final f_final']^T = T[f_initial f_initial']^T
    T = final_TF*initial_TF'

    if abs(sum(T*T')-2*N^2) > 0.2
        println("Warning: T is not unitary")
        mark_with_x = true 
    end

    X , Y = get_X_and_Y(T)
    M = inv(X)*Y 

    i1 = convert_lattice_vector_to_index(final_flux_site,N)
    j0 = convert_lattice_vector_to_index(initial_flux_site,N)

    hopping_amp = C*(-a_init[i1,:]'*M*b_init[j0,:] + a_init[i1,:]'*b_init[j0,:] + 1) 

    display(C)

    return hopping_amp , mark_with_x


    return hopping_amp
end 

function plot_Heisenberg_hopping_vs_system_size(K,N_max)
    hopping_amp_vec = zeros(1,N_max)
    for N = 4:N_max
        hopping_amp_vec[N] , mark_with_x = calculate_z_Heisenberg_hopping_matrix_element(N,K)
        if mark_with_x == false
            scatter(1/N,hopping_amp_vec[N],color = "blue")
        else
            scatter(1/N,hopping_amp_vec[N],color = "blue",marker = "x")
        end
        display(N)
    end
    xlabel("Inverse of linear system size 1/N")
    ylabel("Heisenberg hopping amplitude")
    if K == 1
        ylabel("AFM Kitaev term")
    elseif K == -1
        ylabel("FM Kitaev term")
    end 
end

function get_M_from_H(H)
    N = Int(sqrt(size(H)[1]/2))
    h = H[1:N^2,1:N^2]
    Delta = H[1:N^2,(1+N^2):2*N^2]

    M = 0.5*(h-Delta)

    return M
end 

function find_flipped_bond_coordinates(H)
    """
    Assumes the Hamiltonian contains only a single flipped bond. 
    Finds the coordinate of the flipped bond site in the form [n1,n2] = n1*a1 + n2*a2
    """

    N = Int(sqrt(size(H)[1]/2))
    h = H[1:N^2,1:N^2]
    Delta = H[1:N^2,(1+N^2):2*N^2]

    M = 0.5*(h-Delta)

    K = sign(sum(M))

    bond_indicies_set = findall(x->x==-K,M)
    
    for bond_indicies in bond_indicies_set
        C_A_index = bond_indicies[1]
        if C_A_index % N == 0 
            C_A_n2 = Int(C_A_index/N -1)
            C_A_n1 = Int(N-1) + C_A_n2
        else
            C_A_n2 = Int(floor(C_A_index/N))
            C_A_n1 = Int(C_A_index - C_A_n2*N -1 + C_A_n2)
        end 

        C_B_index = bond_indicies[2]
        if C_B_index % N == 0 
            C_B_n2 = Int(C_B_index/N -1) 
            C_B_n1 = Int(N-1) + C_B_n2
        else
            C_B_n2 = Int(floor(C_B_index/N))
            C_B_n1 = Int(C_B_index - C_B_n2*N -1 + C_B_n2) 
        end 

        if C_A_index == C_B_index
            bond_flavour = "z"
        elseif C_B_n1 == C_A_n1 
            bond_flavour = "y"
        elseif C_B_n2 == C_A_n2
            bond_flavour = "x"
        end
        
        println("flipped bond at R = $C_A_n1 a1 + $C_A_n2 a2 with flavour $bond_flavour")
    end 
end 

function calculate_yz_Gamma_hopping_amplitude(N,K)

    mark_with_x = false
    initial_flux_site = [Int(round(N/2)),Int(round(N/2))]
    final_flux_site = initial_flux_site + [1,0]
    
    initial_HF = get_HF(N,K,initial_flux_site,"z")
    final_HF = get_HF(N,K,final_flux_site,"y")

    #=
    if det(initial_HF) < 10^(-12)
        println("Warning: det initial_HF is small indicating possible degenerate zero modes")
        mark_with_x = true 
    end
    if det(final_HF) < 10^(-12)
        println("Warning: det final_HF is small indicating degenerate zero modes")
        mark_with_x = true 
    end 
    =#

    _ , initial_TF = diagonalise(initial_HF)
    _ , final_TF = diagonalise(final_HF)

    C = get_normalisation_constant(initial_TF,final_TF)

    initial_X , initial_Y = get_X_and_Y(initial_TF)

    a_init = transpose(initial_X + initial_Y)
    b_init = transpose(initial_X - initial_Y)


    # T is a matrix which transforms the creation and annihilation operators for the initial flux sector into those of the final flux sector [f_final f_final']^T = T[f_initial f_initial']^T
    T = final_TF*initial_TF'

    if abs(sum(T*T')-2*N^2) > 0.2
        println("Warning: T is not unitary")
        mark_with_x = true 
    end

    X , Y = get_X_and_Y(T)
    M = inv(X)*Y 

    i1 = convert_lattice_vector_to_index(final_flux_site,N)
    j0 = convert_lattice_vector_to_index(initial_flux_site,N)

    hopping_amp = C*(-a_init[i1,:]'*M*b_init[j0,:] + a_init[i1,:]'*b_init[j0,:] - 1) 

    return hopping_amp , mark_with_x
end 

function convert_lattice_vector_to_index(lattice_vector,N)
    index = 1 + lattice_vector[1] - lattice_vector[2] + N*lattice_vector[2]
    return index
end 

function plot_Gamma_hopping_vs_system_size(K,N_max)
    hopping_amp_vec = zeros(1,N_max)
    for N = 4:N_max
        hopping_amp_vec[N] , mark_with_x = calculate_yz_Gamma_hopping_amplitude(N,K)
        if mark_with_x == false
            scatter(1/N,hopping_amp_vec[N],color = "blue")
        else
            scatter(1/N,hopping_amp_vec[N],color = "blue",marker = "x")
        end
        display(N)
    end
    xlabel("Inverse of linear system size 1/N")
    ylabel("Gamma hopping amplitude")
    if K == 1
        ylabel("AFM Kitaev term")
    elseif K == -1
        ylabel("FM Kitaev term")
    end 
end

function flip_bond_variable(H,bond_site,bond_flavour)
    N = Int(sqrt(size(H)[1]/2))

    M = get_M_from_H(H)

    C_A_index = 1 + bond_site[1] - bond_site[2] + N*bond_site[2]

    if bond_flavour == "z"
        C_B_index = C_A_index
    elseif bond_flavour == "y"
        if bond_site[2] == N - 1
            if bond_site[1] == bond_site[2]
                C_B_index = N
            else
                C_B_index = bond_site[1]-bond_site[2]
            end 
        elseif bond_site[1] == bond_site[2]
            C_B_index = C_A_index + 2*N - 1
        else
            C_B_index = C_A_index + N - 1 
        end 
    else
        if bond_site[1] == bond_site[2] 
            C_B_index = C_A_index + N -1
        else 
            C_B_index = C_A_index -1 
        end 
    end 

    M[C_A_index,C_B_index] = - M[C_A_index,C_B_index]

    H = zeros(2*N^2,2*N^2)

    H[1:N^2,1:N^2] = M + transpose(M)
    H[(N^2+1):2*N^2,1:N^2] = M - transpose(M)
    H[1:N^2,(N^2+1):2*N^2] = transpose(M) - M 
    H[(N^2+1):2*N^2,(N^2+1):2*N^2] = -M - transpose(M) 

    return H 
end 

function calculate_z_Heisenberg_hopping_matrix_element_2(N,K)

    mark_with_x = false
    initial_flux_site = [Int(round(N/2)),Int(round(N/2))]
    final_flux_site = initial_flux_site + [1,0]
    
    initial_HF = get_HF(N,K,initial_flux_site,"z")
    final_HF = get_HF(N,K,final_flux_site,"z")

    _ , initial_TF = diagonalise(initial_HF)
    _ , final_TF = diagonalise(final_HF)

    T = final_TF*initial_TF'

    C = get_normalisation_constant(initial_TF,final_TF)

    X , Y = get_X_and_Y(T)
    M = inv(X)*Y 

    if abs(sum(T*T')-2*N^2) > 0.2
        println("Warning: T is not unitary")
        mark_with_x = true 
    end

    i1 = convert_lattice_vector_to_index(final_flux_site,N)
    j0 = convert_lattice_vector_to_index(initial_flux_site,N)

    op_list = ["cA","cB"]
    id_list = [i1,j0]

    display(C)

    op_dict = form_operator_dictionary(T,initial_TF)

    hopping_amp = C + two_fermion_matrix_element(op_list,id_list,op_dict,M,C) 

    return hopping_amp , mark_with_x
end 

function calculate_yz_Gamma_hopping_amplitude_2(N,K)

    mark_with_x = false
    initial_flux_site = [Int(round(N/2)),Int(round(N/2))]
    final_flux_site = initial_flux_site + [1,0]
    
    initial_HF = get_HF(N,K,initial_flux_site,"z")
    final_HF = get_HF(N,K,final_flux_site,"y")

    _ , initial_TF = diagonalise(initial_HF)
    _ , final_TF = diagonalise(final_HF)

    C = get_normalisation_constant(initial_TF,final_TF)

    # T is a matrix which transforms the creation and annihilation operators for the initial flux sector into those of the final flux sector [f_final f_final']^T = T[f_initial f_initial']^T
    T = final_TF*initial_TF'

    if abs(sum(T*T')-2*N^2) > 0.2
        println("Warning: T is not unitary")
        mark_with_x = true 
    end

    X , Y = get_X_and_Y(T)
    M = inv(X)*Y 

    i1 = convert_lattice_vector_to_index(final_flux_site,N)
    j0 = convert_lattice_vector_to_index(initial_flux_site,N)

    op_dict = form_operator_dictionary(T,initial_TF)

    hopping_amp = two_fermion_matrix_element(["cA","cB"],[i1,j0],op_dict,M,C) - C

    return hopping_amp , mark_with_x
end 

function form_operator_dictionary(T,T_u1)
    """
    4 types of operators:
    - "f^u2" a destruction operator for the left vacuum
    - "cA" an A site Majorana fermion
    - "cB" a B site Majorana fermion
    -  "f'" a creation operator for the right vacuum

    op_dict gives the matricies needed to express an operator in terms of creation and annihilation opertaors for the left vacuum in a matrix element. 
    """
    N = Int(sqrt(size(T)[1]/2))

    X , Y = get_X_and_Y(T)
    X_u1 , Y_u1 = get_X_and_Y(T_u1)

    a = transpose(X_u1 + Y_u1)
    b = transpose(X_u1 - Y_u1)

    op_dict = Dict()
    op_dict["f^u2"] = [conj(X),conj(Y)]
    op_dict["cA"] = [a , conj(a)]
    op_dict["cB"] = [b , -conj(b)]
    op_dict["f'"] = [zeros(N^2,N^2),Matrix(I,N^2,N^2)] 
   
    return op_dict 
end

function contract(operator_1,index_1,operator_2,index_2, op_dict, M)
    """
    Calculates a generalised contraction of two operators between two different vacua <0^u2|psi_1 psi_2 |0^u1> 
    """
    x1 , y1 = op_dict[operator_1][1] , op_dict[operator_1][2]
    x2 , y2 = op_dict[operator_2][1] , op_dict[operator_2][2]

    return  x1[index_1,:]'*y2[index_2,:] - y1[index_1,:]'*M*y2[index_2,:]
end 

function four_fermion_matrix_element(op_list,id_list,op_dict,M,C)

    matrix_element = (contract(op_list[1],id_list[1],op_list[2],id_list[2],op_dict,M))*contract(op_list[3],id_list[3],op_list[4],id_list[4],op_dict,M)
    - contract(op_list[1],id_list[1],op_list[3],id_list[3],op_dict,M)*contract(op_list[2],id_list[2],op_list[4],id_list[4],op_dict,M)
    + contract(op_list[1],id_list[1],op_list[4],id_list[4],op_dict,M)*contract(op_list[2],id_list[2],op_list[3],id_list[3],op_dict,M)

    matrix_element = matrix_element*C

    return matrix_element
end

function two_fermion_matrix_element(op_list,id_list,op_dict,M,C)
    matrix_element = C*contract(op_list[1],id_list[1],op_list[2],id_list[2],op_dict,M)
    return matrix_element
end

function aligned_flux_pair_creation_amplitude(n,m,op_dict,M,C)

    N = Int(sqrt(size(M)[1]))
    i0 = Int(convert_lattice_vector_to_index([round(N/2),N],N))
    
    amplitude = -two_fermion_matrix_element(["cA","f'"],[i0,n],op_dict,M,C)*two_fermion_matrix_element(["cA","f'"],[i0,m],op_dict,M,C)
    -two_fermion_matrix_element(["cB","f'"],[i0,n],op_dict,M,C)*two_fermion_matrix_element(["cB","f'"],[i0,m],op_dict,M,C)

    return amplitude
end 

function aligned_flux_pair_hopping_amplitude(n,m,k,l,op_dict,M,C)

    N = Int(sqrt(size(M)[1]))
    j0 = Int(convert_lattice_vector_to_index([round(N/2),N],N))
    i2 = Int(convert_lattice_vector_to_index([round(N/2),N-1],N))

    amplitude = - four_fermion_matrix_element(["f^u2","cA","cB","f'"],[m,i2,j0,l],op_dict,M,C)*four_fermion_matrix_element(["f^u2","cA","cB","f'"],[n,i2,j0,k],op_dict,M,C)
    - two_fermion_matrix_element(["f^u2","f'"],[m,l],op_dict,M,C)two_fermion_matrix_element(["f^u2","f'"],[n,k],op_dict,M,C)

    return amplitude
end 

function get_Kagome_Hamiltonian(k,hop_amp)

    phi_z = exp(im*dot(k,nn[1]-nn[2]))
    phi_y = exp(im*dot(k,nn[1]-nn[3]))
    phi_x = exp(im*dot(k,nn[2]-nn[3]))


    H_kagome = [0 hop_amp*(1+phi_z) hop_amp*(1+phi_y);
    hop_amp*(1+phi_z') 0 hop_amp*(1+phi_x);
    hop_amp*(1+phi_y') hop_amp*(1+phi_x') 0]

    return Hermitian(H_kagome)
end

Num_states = 7
Energies = LinRange(0,0.1,Num_states)
hopping_amplitude_matrix = 0.005*randn(Num_states,Num_states,Num_states,Num_states)

for i = 1:Num_states
    for j = 1:Num_states
        for k = 1:Num_states
            for l = 1:Num_states
                hopping_amplitude_matrix[j,i,l,k]=hopping_amplitude_matrix[i,j,k,l]
                hopping_amplitude_matrix[k,l,i,j]=hopping_amplitude_matrix[i,j,k,l]'
                hopping_amplitude_matrix[l,k,j,i]=hopping_amplitude_matrix[i,j,k,l]'
            end 
        end 
    end 
end 

function get_full_effective_Hamiltonian(k,Energies,hopping_amplitude_matrix)
    N = Int(size(Energies)[1])
    num_bands = 3*N^2
    H_full = zeros(Complex{Float64},num_bands,num_bands)

    Delta = 0.26 

    for n = 1:Int(num_bands/3)
        for m = 1:Int(num_bands/3)
            layer_1_final_state = Int(floor((n-1)/N))+1 
            layer_2_final_state = n - N*(layer_1_final_state-1)
            layer_1_initial_state = Int(floor((m-1)/N))+1 
            layer_2_initial_state = m - N*(layer_1_initial_state-1)
            H_full[(1+(n-1)*3):n*3,(1+(m-1)*3):m*3] = get_Kagome_Hamiltonian(k,hopping_amplitude_matrix[layer_1_final_state,layer_2_final_state,layer_1_initial_state,layer_2_initial_state])

            if n == m 
                for i = 1:3
                    H_full[(3*(n-1)+i),(3*(m-1)+i)] = Delta + Energies[layer_1_initial_state] + Energies[layer_2_final_state]
                end
            end 
        end 
    end
    return Hermitian(H_full)
end 

function get_k_GtoKtoMtoG(Num_points)
    """
    Used to create a list of k vectors along the path G to K to M to G in the Brillouin zone. 
    Takes:
    - Num_points which is the number of k vectors to sample along the path

    returns:
    - a 2xNum_points matrix, where each collumn is a k vector
    """
    
    # This section is needed to ensure that k points are sampled evenly along the path 
    Num_points_GtoKtoM = Int(round(Num_points*3/(3+sqrt(3))))
    Num_points_MtoG = Int(round(Num_points*sqrt(3)/(3+sqrt(3)))) 
    Num_points = Num_points_GtoKtoM + Num_points_MtoG

    kGtoKtoMtoG = zeros(2,Num_points)
    [kGtoKtoMtoG[1,i] = g1[1]*i/Num_points_GtoKtoM for i = 1:Num_points_GtoKtoM]
    [kGtoKtoMtoG[2,Num_points_GtoKtoM+i] = g1[2]*(1-i/Num_points_MtoG) for i = 1:Num_points_MtoG]
    return(kGtoKtoMtoG)
end

function get_bandstructure_GtoKtoMtoG(kGtoKtoMtoG,Energies,hopping_amplitude_matrix)
    """
    A faster alternative way to calculate the bandstructure along G to K to M to G than using get_bilayer_bandstructure.
    Rather than calculating the bandstructure of the entire Brillouin zone then finding k vectors along the path, this only calculates the energies at k vectors which lie on the path you want to plot.
    This takes:
    - a prepared list of k vectors along the path G to K to M to G as a 2xN matrix, N being the number of points sampled along the path 
    """

    Num_k_points = Int(size(kGtoKtoMtoG)[2])

    bands_GtoKtoMtoG = zeros(3*Int(size(Energies)[1]^2),Num_k_points)
    
    for i = 1:Num_k_points
        k = kGtoKtoMtoG[:,i]
        H = get_full_effective_Hamiltonian(k,Energies,hopping_amplitude_matrix)
        bands_GtoKtoMtoG[:,i] = eigvals(H)
    end

    return bands_GtoKtoMtoG
end

function plot_bands_GtoKtoMtoG(kGtoKtoMtoG,bands_GtoKtoMtoG,axes=gca())
    """
    An alternative way to plot the bandstructure (Faster than plot_bands_G_to_K_to_M_to_G) that takes:
    - kGtoKtoMtoG a prepared list of k vectors along the path G to K to M to G as a 2xN matrix where N is the number of k vectors sampled along the path
    - bands_GtoKtoMtoG a list of energies at the corresponding k vectors as a 16xN matrix
    - an axis object axes used to add features to the plot such as labels. 
    """
    Num_k_points = Int(size(kGtoKtoMtoG)[2])

    K_index = round(2*Num_k_points/(3+sqrt(3)))
    M_index = round(3*Num_k_points/(3+sqrt(3)))
    
    
    E_max = 0 
    for i = 1:16 
        if E_max < maximum(bands_GtoKtoMtoG[i,:]) 
            E_max = maximum(bands_GtoKtoMtoG[i,:])
        end
        axes.plot(1:Num_k_points,bands_GtoKtoMtoG[i,:],color="black")
    end
    

    axes.set_xticks([0,K_index,M_index,Num_k_points])
    axes.set_xticklabels(["\$\\Gamma\$","K","M","\$\\Gamma\$"])
    axes.set_ylabel("Energy")
    axes.vlines([0,K_index,M_index,Num_k_points],-(1.1*E_max),1.1*E_max,linestyle="dashed")
end


#=
This final section of code is an attempt to use the  3 fold rotation symmetry of the ground state to reduce the Hamiltonian to block diagonal form before diagonalising. 
The code is successful at block diagonalising and finding the diagonalising transformation but the transformation is not unitary and is only applicable to the ground state so for now it is left incomplete. 


function form_M_matrix(N)
    diag_block = zeros(N,N)
    for j = 1:(N-1)
        diag_block[j,j]=1
        diag_block[j+1,j]=1
    end
    diag_block[N,N] = 1

    A = zeros(N^2,N^2)
    for j = 1:(N-1)
        A[(1+(j-1)*N):(j*N),(1+(j-1)*N):(j*N)] = diag_block
        A[(1+(j-1)*N):(j*N),(1+(j)*N):((j+1)*N)] = Matrix(I,N,N)
    end 
    A[(1+N^2-N):N^2,(1+N^2-N):N^2] = diag_block

    B = zeros(N^2,N^2)
    C = zeros(N^2,N^2)

    for j = 1:N
        B[N^2-N+j,j*N] = 1
        C[(1+(j-1)*N),j] = 1
    end

    M = [ A B C ; C A B ; B C A]

    return M 
end 

function form_R_matrix(N)
    """
    R is a 3N^2 x 3N^2 matrix which transforms between the lattice sites basis and eigenstates of the C3 rotation symmetry operator. 
    """
    l = exp(2*pi*im/3)
    R = (1/sqrt(3))* [ I(N^2) l*I(N^2) conj(l)*I(N^2) ; I(N^2) conj(l)*I(N^2) l*I(N^2) ; I(N^2) I(N^2) I(N^2)]

    return R 
end 

function form_H0_matrix(N)
    """
    Calculates a matrix Hamiltonian for the fluxless Kitaev model:
    H = 1/2 (f' f) H (f f')^T
    where f are complex matter fermions 
    """
    M = form_M_matrix(N)
    h = M + M'
    Delta = M'- M 
    
    return [h Delta ; Delta' -h]
end 

function form_H_prime(N)
    """
    Calculates a matrix Hamiltonian for the fluxless Kitaev model in the rotated basis of symmetry eigenstates:
    H = 1/2 (F' F) H' (F F')^T
    where F are complex matter fermions in the C3 symmetry eigenbasis 
    """
    R = form_R_matrix(N)
    M = form_M_matrix(N)

    curly_M = R*M*R'

    h = curly_M + curly_M'
    Delta = curly_M' - curly_M

    return [ h Delta ; Delta' -h ]
end 

function form_gamma_matrix(N)
    gamma = zeros(6*N^2,6*N^2)
    gamma[1:N^2,1:N^2] = I(N^2)
    gamma[(N^2+1):2*N^2,(1+3*N^2):4*N^2] = I(N^2)
    gamma[(2*N^2+1):3*N^2,(1+N^2):2*N^2] = I(N^2)
    gamma[(3*N^2+1):4*N^2,(4*N^2+1):5*N^2] = I(N^2)
    gamma[(4*N^2+1):5*N^2,(2*N^2+1):3*N^2] = I(N^2)
    gamma[(5*N^2+1):6*N^2,(5*N^2+1):6*N^2] = I(N^2) 

    return gamma
end 

function form_gamma_2_matrix(N)
    gamma = zeros(6*N^2,6*N^2)
    reverse_order = zeros(N^2,N^2)
    for j = 1:N^2
        reverse_order[N^2+1-j,j] = 1
    end 
    
    gamma[1:N^2,(N^2+1):2*N^2] = I(N^2)
    gamma[(N^2+1):2*N^2,(3*N^2+1):4*N^2] = I(N^2)
    gamma[(2*N^2+1):3*N^2,(5*N^2+1):6*N^2] = I(N^2)
    gamma[(3*N^2+1):4*N^2,1:N^2] = reverse_order
    gamma[(4*N^2+1):5*N^2,(2*N^2+1):3*N^2] = reverse_order
    gamma[(5*N^2+1):6*N^2,(4*N^2+1):5*N^2] = reverse_order

    return gamma
end


function block_diagonalise(H_prime)
    N = Int(sqrt(size(H_prime)[1]/6))
    g = form_gamma_matrix(N)
    
    return g*H_prime*g'
end 

function diagonalise_blocks(H_prime)
    """
    This diagonalises the matrix H_prime in a single function. It works by: 
    - transforming H_prime into block diagonal form with 3 2N^2 x 2N^2 blocks 
    - Diagonalising each block seperately and forming a block diagonal unitary matrix U 
    - Transforming U back into the original basis, with reordered eigenvalues 
    returns 
    - T_prime, a unitary matrix such that T_prime*H_prime*T_prime' = [E 0 ; -E 0]

    NOTE the eigenvalue in E are NOT in increasing size order. 
    """
    N = Int(sqrt(size(H_prime)[1]/6))
    g = form_gamma_matrix(N)
    g2 = form_gamma_2_matrix(N)

    block_diagonal_H = g*H_prime*g'

    U = zeros(Complex{Float64},6*N^2,6*N^2)

    for j = 1:3
        U[(1+2*(j-1)*N^2):j*2*N^2,(1+2*(j-1)*N^2):j*2*N^2] = eigvecs(block_diagonal_H[(1+2*(j-1)*N^2):j*2*N^2,(1+2*(j-1)*N^2):j*2*N^2])
    end 

    T_prime = g2*U'*g

    return T_prime
end

function diagonalise_H0(H_prime)
    N = Int(sqrt(size(H_prime)[1]/6))
    g = form_gamma_matrix(N)
    g2 = form_gamma_2_matrix(N)
    R = form_R_matrix(N)

    block_diagonal_H = g*H_prime*g'

    U = zeros(Complex{Float64},6*N^2,6*N^2)

    for j = 1:3
        U[(1+2*(j-1)*N^2):j*2*N^2,(1+2*(j-1)*N^2):j*2*N^2] = eigvecs(block_diagonal_H[(1+2*(j-1)*N^2):j*2*N^2,(1+2*(j-1)*N^2):j*2*N^2])
    end 

    T_prime = g2*U'*g

    T = T_prime*[ R zeros(3*N^2,3*N^2) ; zeros(3*N^2,3*N^2) R]

    Y = T[(3N^2+1):end,1:3*N^2]
    X = T[(3N^2+1):end,(3N^2+1):end]

    return [conj(X) conj(Y) ; Y X]
end 
=#