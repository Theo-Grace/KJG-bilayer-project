using LinearAlgebra
using SparseArrays
using Arpack 
using PyPlot
pygui(true) 

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

function diagonalise(H)
    """
    Diagonalises a given Hamiltonian H. H is assumed to be real and symmetric.
    returns:
    - a full spectrum of eigenvalues. It is assumed that for each energy E, -E is also an eigenvalue.
    - A unitary (orthogonal as T is real) matrix T which diagonalises H
    """
    N_sq = Int(size(H)[1]/2)
    half_spectrum , eigvecs = eigs(H, nev = N_sq, which =:LR)

    energies = zeros(2*N_sq,1)
    energies[1:N_sq] = half_spectrum
    energies[N_sq+1:end] = - half_spectrum

    X = eigvecs[1:N_sq,1:N_sq]'
    Y = eigvecs[(N_sq+1):2*N_sq,1:N_sq]'

    T = [X Y ; Y X]

    return energies , T 
end 

function get_HF(N,K=1,flux_site=[Int(round(N/2)),Int(round(N/2))],flux_flavour="z")
    """
    Creates a 2N^2 x 2N^2 Hamiltonian matrix for a flux sector with a single flux pair 
    The flux pair is located at the lattice site flux_site which is given as a coordinate [n1,n2] = n1*a1 + n2*a2 where a1,a2 are plvs.
    flux_flavour specifies the type of bond which connects the fluxes in the pair. It is given as a string, either "x","y" or "z"
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
    B[N,N] = K

    M = spzeros(N^2,N^2)
    for j = 1:(N-1)
        M[(1+(j-1)*N):(j*N),(1+(j-1)*N):(j*N)] = A
        M[(1+(j-1)*N):(j*N),(1+j*N):((j+1)*N)] = B
    end

    M[N*(N-1)+1:N^2,N*(N-1)+1:N^2] = A
    M[N*(N-1)+1:N^2,1:N] = B

    if flux_flavour == "x"
        B[flux_site[1],flux_site[1]] = -K
        M[(1+(flux_site[2]-1)*N):(flux_site[2]*N),(1+flux_site[2]*N):((flux_site[2]+1)*N)] = B
        display("BF is")
        display(B)
    elseif flux_flavour == "y"
        if flux_site[1] == 1
            A[1,N] = -K
        else
            A[flux_site[2],flux_site[2]-1] = -K
        end 
    else
        A[flux_site[1],flux_site[1]] = -K

    end 
    
    M[(1+(flux_site[2]-1)*N):(flux_site[2]*N),(1+(flux_site[2]-1)*N):(flux_site[2]*N)] = A
        
    H = spzeros(2*N^2,2*N^2)

    H[1:N^2,1:N^2] = M + transpose(M)
    H[(N^2+1):2*N^2,1:N^2] = M - transpose(M)
    H[1:N^2,(N^2+1):2*N^2] = transpose(M) - M 
    H[(N^2+1):2*N^2,(N^2+1):2*N^2] = -M - transpose(M) 

    return H 

end 

function get_X_and_Y(T)
    """
    Given a unitary matrix T this function extracts the X and Y matrices 
    """
    N_sq = Int(size(T)[1]/2)
    X = T[(N_sq+1):2*N_sq,(N_sq+1):2*N_sq]
    Y = T[(N_sq+1):2*N_sq,1:N_sq]

    return X , Y 
end 

function transform_between_flux_sectors(TF1,TF2)
    """
    Given the Hamiltonian for two flux sectors F1 and F2, this calculates unitary transformation T. 
    Applying T to the creation and annihilation operators of F1 gives the corresponding operators for F2
    """
    T = TF2*TF1'

    return T 
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

function calculate_z_Heisenberg_hopping_matrix_element(N,K,initial_flux_site=[1,1])
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

    display(b'*F*a)

    hopping_amp = C*( a'*b - b'*F*a + 1)

    return hopping_amp
end 


