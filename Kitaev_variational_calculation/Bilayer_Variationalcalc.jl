# This is an updated version of the variational calculation for the bilayer Kitaev model

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

function lattice(a1,a2,N)
    return [[i,j] for i=0:(N-1), j = 0:(N-1)]
end 

function my_brillouinzone(N)
    BZ = [[n1/N,n2/N] for n1=0:(N-1), n2=0:(N-1)]

    return BZ
end 

# This section calculates the dual vectors and makes a matrix of vectors in the Brillouin zone
g1, g2 = dual(a1,a2)

# This section includes functions to visualise the geometry of the lattice and BZ used in calculations 

function plot_my_lattice(N)
    RL = lattice(a1,a2,N)

    for vec in RL
        r = vec[1]*a1+vec[2]*a2

        r_B = r+nz

        scatter(r[1],r[2],color="r")
        scatter(r_B[1],r_B[2],color="b")
    end 
    scatter(a1[1],a1[2])
    scatter(a2[1],a2[2])
end 

function plot_my_BZ(N)
    BZ = my_brillouinzone(N)

    for k_vec in BZ
        k = k_vec[1]*g1+k_vec[2]*g2

        scatter(k[1],k[2],color="b")
    end 
    scatter(g1[1],g1[2])
    scatter(g2[1],g2[2])
end

function plot_exact_spectrum(N)
    for n = 0.1*(1:N)
        scatter(n,get_exact_GS_matter_fermion_energy([n,0]))
    end 
end


# This section is used to create Hamiltonians and uses the numbering convention index = 1 + n1 + n2*N with:
# a1 = [1/2, sqrt(3)/2]
# a2 = [-1/2, sqrt(3)/2]


function get_H0(N,K=1)
    """
    Calculates a 2N^2 x 2N^2 matrix H0 which is the Hamiltonian for a flux free Kitaev model in terms of complex matter fermions. 
    - K is the Kitaev parameter (take to be +1 or -1)
    returns:
    - a 2N^2 x 2N^2 Hamiltonian matrix which is real and symmetric 

    uses the lattice vectors 
    a1 = [1/2, sqrt(3)/2]
    a2 = [-1/2, sqrt(3)/2]

    and numbering convention i = 1 + n1 + N*n2 to represent a site [n1,n2] = n1*a1 + n2*a2 
    """
    A = zeros(N,N)
    B = zeros(N,N)

    for j = 1:N-1
        A[j,j] = K
        A[j+1,j] = K
        B[j,j] = K
    end 
    A[N,N] = K
    A[1,N] = K
    B[N,N] = K

    M = zeros(N^2,N^2)
    for j = 1:(N-1)
        M[(1+(j-1)*N):(j*N),(1+(j-1)*N):(j*N)] = A
        M[(1+j*N):((j+1)*N),(1+(j-1)*N):(j*N)] = B
    end

    M[N*(N-1)+1:N^2,N*(N-1)+1:N^2] = A
    M[1:N,N*(N-1)+1:N^2] = B

    H = zeros(2*N^2,2*N^2)

    H[1:N^2,1:N^2] = M + transpose(M)
    H[(N^2+1):2*N^2,1:N^2] = M - transpose(M)
    H[1:N^2,(N^2+1):2*N^2] = transpose(M) - M 
    H[(N^2+1):2*N^2,(N^2+1):2*N^2] = -M - transpose(M) 

    return -H
end

function get_M_from_H(H)
    """
    Given a matrix H such that the Hamiltonian is  1/2 [f' f]H[f f']^T this extracts a matrix M which is used to express the Hamiltonian as i/2 [cA cB][0 M; -M^T 0][cA cB]
    """
    N = Int(sqrt(size(H)[1]/2))
    h = H[1:N^2,1:N^2]
    Delta = H[1:N^2,(1+N^2):2*N^2]

    M = 0.5*(h-Delta)

    return M
end 

function flip_bond_variable(H,bond_site,bond_flavour)
    """
    Given a Hamiltonian H this returns a new Hamiltonian with a reversed sign for the bond variable at site bond_site with orientation bond_flavour  
    """
    N = Int(sqrt(size(H)[1]/2))

    M = get_M_from_H(H)

    C_A_index = 1 + bond_site[1] + N*bond_site[2]

    if bond_flavour == "z"
        C_B_index = C_A_index
    elseif bond_flavour == "y"
        if bond_site[2] == 0
            C_B_index = N*(N-1) + C_A_index
        else
            C_B_index = C_A_index - N 
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
    """
    Given a lattice vector in the form [n1,n2] this assigns an integer according to a specific numbering convention
    """
    index = 1 + lattice_vector[1] + N*lattice_vector[2]
    return Int(index)
end 

function convert_index_to_lattice_vector(index,N)
    if index % N == 0 
        n1 = Int(N-1)
        n2 = Int(index/N -1)
    else
        n2 = Int(floor(index/N))
        n1 = Int(index - n2*N -1)
    end 

    return [n1,n2]
end 

function find_flipped_bond_coordinates(H)
    """
    Finds the coordinate of the flipped bond site in the form [n1,n2] = n1*a1 + n2*a2
    """

    N = Int(sqrt(size(H)[1]/2))

    M = get_M_from_H(H)

    K = sign(sum(M))

    bond_indicies_set = findall(x->x==-K,M)
    
    for bond_indicies in bond_indicies_set
        C_A_index = bond_indicies[1]
        
        C_A_vec = convert_index_to_lattice_vector(C_A_index,N)

        C_B_index = bond_indicies[2]

        C_B_vec = convert_index_to_lattice_vector(C_B_index,N)

        if C_A_index == C_B_index
            bond_flavour = "z"
        elseif C_B_vec[1] == C_A_vec[1]
            bond_flavour = "y"
        elseif C_B_vec[2] == C_A_vec[2]
            bond_flavour = "x"
        end
        
        println("flipped bond at R = $(C_A_vec[1]) a1 + $(C_A_vec[2]) a2 with flavour $bond_flavour")
    end 
end 

# This section includes functions to diagonalise Hamiltonians numerically and find transformations between flux sectors 

function diagonalise(H)
    """
    returns:
    - Eigenvalues of the Hamiltonian as a vector 
    - A unitary matrix T formed from eigenvectors on the rows such that T*H*T' = [E 0 ; 0 -E]
    The matrix T has the form: 

    T = [X* Y* ;
         Y  X ]
    """
    N_sq = Int(size(H)[1]/2)

    H = Matrix(H)

    energies = eigvals(H)
    reverse!(energies)
    eigvec = eigvecs(H)

    gamma = [zeros(N_sq,N_sq) Matrix(I,N_sq,N_sq) ; Matrix(I,N_sq,N_sq) zeros(N_sq,N_sq)] # This reverses the negative eigenvalues so the diagonal matrix has the correct form. 
    eigvec = gamma*eigvec

    X = eigvec[1:N_sq,1:N_sq]'
    Y = eigvec[(N_sq+1):2*N_sq,1:N_sq]'

    T = [conj(X)  conj(Y) ; Y X]

    return energies , T 
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

#This section contains functions to efficiently calculate Matrix elements  

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
    """
    Calculates a matrix element of four fermionic operators between two different vacua
    takes:
    - op_list a list of operator types given as a vector of strings. The possible choices are: 
        4 types of operators:
        - "f^u2" a destruction operator for the left vacuum
        - "cA" an A site Majorana fermion
        - "cB" a B site Majorana fermion
        -  "f'" a creation operator for the right vacuum

    - id_list a vector of 4 integers specifying the index associated to the operator. The order of the indices is the same as the order of the operators in the list
    - op_dict a dictionary which relates operator types to the matrices used to calculate the matrix element
    - M   M = inv(X)*Y where X and Y are calculated from the transformation T relating the left and right vacua
    - C the overlap between the left and right vacua 
    """

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

# This section contains functions for visualising eigenstates 
function calculate_number_density_at_site(site,T_u,ex_id)
    """
    Calculates the complex matter fermion density at the given lattice site 
    f_r = cA_r + icB_r

    takes:
    - A lattice vector given as [n1,n2]
    - T_u which is a unitary transformation which diagonalises a flux sector Hamiltonian
    - ex_id which is an integer specifying which excited state to calculate the density for.  
    """
    X , Y = get_X_and_Y(T_u)

    N = Int(sqrt(size(X)[1]))
    i0 = convert_lattice_vector_to_index(site,N)
    
    
    number_density =  X'[i0,ex_id]*X'[i0,ex_id] - Y'[i0,ex_id]*Y'[i0,ex_id] #+ sum(Y'[i0,:].^2)

    #display(dot(Y[i0,:],conj.(Y[i0,:])))
    return number_density
end 

function plot_eigenstate_fermion_density(T,ex_id,size_scale=10,colour="b")
    N = Int(sqrt(size(T)[1]/2))
    lattice = zeros(2,N^2)
    num_density_vector = zeros(1,N^2)

    excitation_number = N^2+1 -ex_id

    for i = 1:N^2
        site = convert_index_to_lattice_vector(i,N)
        num_density = calculate_number_density_at_site(site,T,excitation_number)

        real_space_coords = site[1]*a1 + site[2]*a2

        lattice[:,i] = real_space_coords

        num_density_vector[1,i] = num_density
    end

    scatter(lattice[1,:],lattice[2,:],sizes = size_scale*num_density_vector[1,:],color=colour)
    scatter(lattice[1,:].+N*a1[1],lattice[2,:].+N*a1[2],sizes = size_scale*num_density_vector[1,:],color=colour,alpha=0.5)
    scatter(lattice[1,:].+N*a2[1],lattice[2,:].+N*a2[2],sizes = size_scale*num_density_vector[1,:],color=colour,alpha=0.5)
    scatter(lattice[1,:].+N*(a1[1]+a2[1]),lattice[2,:].+N*(a1[2]+a2[2]),sizes = size_scale*num_density_vector[1,:],color=colour)
    scatter(lattice[1,:].-N*a2[1],lattice[2,:].-N*a2[2],sizes = size_scale*num_density_vector[1,:],color=colour,alpha=0.5)
    scatter(lattice[1,:].-N*a1[1],lattice[2,:].-N*a1[2],sizes = size_scale*num_density_vector[1,:],color=colour,alpha=0.5)

    xlim(0,2*N*a1[1])
    ylim(-N*a1[2],N*a1[2])

    if ex_id%10 == 1 
        th = "st"
    elseif ex_id%10 ==2
        th = "nd"
    elseif ex_id%10 ==3
        th = "rd"
    else
        th ="th"
    end

    title("Number density plot for $ex_id"*"$th Excited state")
end 


# This adds functions to find the effective Hamiltonian 
function get_Kagome_Hamiltonian(k,hop_amp)

    phi_z = exp(im*dot(k,nn[1]-nn[2]))
    phi_y = exp(im*dot(k,nn[1]-nn[3]))
    phi_x = exp(im*dot(k,nn[2]-nn[3]))


    H_kagome = [0 hop_amp*(1+phi_z) hop_amp*(1+phi_y);
    hop_amp*(1+phi_z') 0 hop_amp*(1+phi_x);
    hop_amp*(1+phi_y') hop_amp*(1+phi_x') 0]

    return Hermitian(H_kagome)
end

function get_full_isolated_flux_effective_Hamiltonian(Q_vec,delta_k_vec,hopping_amplitude,TH)
    Delta = 0.26 
    N = Int(sqrt(size(TH)[1]))

    Q = g1*Q_vec[1] + g2*Q_vec[2]
    delta_k = g1*delta_k_vec[1] + g2*delta_k_vec[2]

    H_eff = zeros(Complex{Float64},4,4)
    H_eff[1:3,1:3] = get_Kagome_Hamiltonian(Q,hopping_amplitude)
    H_eff[1:3,1:3] += 2*Delta*Matrix(I,3,3)

    k_vec = shift_k_to_BZ((Q_vec./2 +delta_k_vec),N)
    q_vec = shift_k_to_BZ((Q_vec./2 -delta_k_vec),N)

    k_id = convert_lattice_vector_to_index(Int.(round.(k_vec)),N)
    q_id = convert_lattice_vector_to_index(Int.(round.(q_vec)),N)

    H_eff[1:3,4] = TH[k_id,q_id,:]

    H_eff[4,4] = get_exact_GS_matter_fermion_energy(Q./2+delta_k)+get_exact_GS_matter_fermion_energy(Q./2-delta_k)

    return Hermitian(H_eff)
end 

# This section contains functions to calculate exact eigenstates of the fluxless Kitaev Hamiltonian 
function phase_factor(k,K)
    """
    Calculates the phase factor Arg( Sum_alpha (e^(ik . r_alpha))) used to calculate exact eigenstates of the fluxless Kitaev model 
    """
    phase = 0 

    for nn_vector in nn
        phase += K*exp(im*dot(k,nn_vector))
    end 
    phase = phase/sqrt(phase*phase')
    return phase
end 

function get_T0_exactly(H0)
    """
    Calculates the exact transformation T0 which diagonalises the fluxless matter sector Hamiltonian T0*H0*T0' = diagonal 
    """
    N = Int(sqrt(size(H0)[1]/2)) 

    K = -sign(sum(get_M_from_H(H0)))

    T_K = zeros(Complex{Float64},2*N^2,2*N^2)

    T_K[1:N^2,1:N^2] = Matrix(I,N^2,N^2)
    T_K[1:N^2,(N^2+1):2*N^2] = Matrix(I,N^2,N^2)

    k_lattice = [[n1/N,n2/N] for n1=0:(N-1), n2=0:(N-1)]

    for k_vectors in k_lattice
        k_id = convert_lattice_vector_to_index(round.(N*k_vectors),N)
        k = (k_vectors[1])*g1 + (k_vectors[2])*g2

        T_K[(N^2+k_id),k_id]=im*phase_factor(k,K)
        T_K[(N^2+k_id),(N^2+k_id)]=-im*phase_factor(k,K)
    end 

    F = get_fourier_matrix(N)

    T0 = ((1/(2)).*[Matrix(I,N^2,N^2) im*Matrix(I,N^2,N^2) ; Matrix(I,N^2,N^2) -im*Matrix(I,N^2,N^2)]*F*T_K)'

    return T0
end 

function get_fourier_matrix(N)
    """
    Calculates a matrix F which can be used to implement a Fourier transform when the Majorana fermions have the form [cA_r cB_r] = F[cA_k cB_k]
    """

    real_lattice = [[n1,n2] for n1=0:(N-1) , n2 = 0:(N-1)]
    k_lattice = [[n1/N,n2/N] for n1=0:(N-1), n2=0:(N-1)]

    fourier_matrix = zeros(Complex{Float64},2*N^2,2*N^2)

    for vectors in real_lattice
        for k_vectors in k_lattice

            R_id = convert_lattice_vector_to_index(vectors,N)
            R = vectors[1]*a1 + vectors[2]*a2

            k_id = convert_lattice_vector_to_index(round.(N*k_vectors),N)
            k = (k_vectors[1])*g1 + (k_vectors[2])*g2

            fourier_matrix[R_id,k_id] = exp(-im*dot(k,R))
            fourier_matrix[N^2+R_id,N^2+k_id] = exp(-im*(dot(k,R+nz)))
        end 
    end 

    fourier_matrix = (1/N)*fourier_matrix

    return fourier_matrix
end 

function get_exact_GS_matter_fermion_energy(k)
    E_comp = 2*(1 + exp(im*dot(k,a1)) + exp(im*dot(k,a2)))

    E = sqrt(E_comp*E_comp')

    return E 
end 

# This section adds functions to calculate matrix elements using exact eigenstates 

function flux_spin_plane_wave_matrix_element(N,K)
    H0 = get_H0(N,K)

    T0 = get_T0_exactly(H0)

    BZ = my_brillouinzone(N)

    A_matrix_elements = zeros(Complex{Float64},N^2,3)
    B_matrix_elements = zeros(Complex{Float64},N^2,3)

    for (j,flavour) in enumerate(["x","y","z"])

        HF = flip_bond_variable(H0,[1,1],flavour)
        _,TF = diagonalise(HF)

        T = TF*T0'

        if abs(sum(T*T')-2*N^2) > 0.2
            println("Warning: T is not unitary")
            mark_with_x = true 
        end

        op_dict = form_operator_dictionary(T,T0)

        X,Y = get_X_and_Y(T)
        M = inv(X)*Y
        C = (det(X'*X))^(1/4)
        
        for k_vec in BZ
            k_id = convert_lattice_vector_to_index(Int.(round.(N*k_vec)),N)

            A_matrix_elements[k_id,j] = im*two_fermion_matrix_element(["cA","f'"],[1,k_id],op_dict,M,C)
            B_matrix_elements[k_id,j] = im*two_fermion_matrix_element(["cB","f'"],[1,k_id],op_dict,M,C)
        end 
    end 

    return N*A_matrix_elements, N*B_matrix_elements
end 

function calculate_hybridisation_amplitudes_TH(N,K)
    A_ME, B_ME = flux_spin_plane_wave_matrix_element(N,K)

    BZ = my_brillouinzone(N)

    T_H = zeros(Complex{Float64},N^2,N^2,3)

    for k in BZ
        for q in BZ

            k_id = convert_lattice_vector_to_index(Int.(round.(N*k)),N)
            q_id = convert_lattice_vector_to_index(Int.(round.(N*q)),N)

            for flavour = [1,2,3]
    
                T_H[k_id,q_id,flavour] = A_ME[k_id,flavour]*A_ME[q_id,flavour]+B_ME[k_id,flavour]*B_ME[q_id,flavour]
            end
        end 
    end 

    return T_H
end 

function get_bandstructure(BZ,hop_amp,TH)
    N = Int(sqrt(size(TH)[1]))

    bandstructure = Dict()
    H_eff = zeros(Complex{Float64},4,4)

    for (n,Q_vec) in enumerate(BZ)
        bandstructure[Q_vec] = []
        for delta_k_vec in BZ
        
            H_eff = get_full_isolated_flux_effective_Hamiltonian(Q_vec,delta_k_vec,hop_amp,TH)
            append!(bandstructure[Q_vec],eigvals(H_eff))
        end
        display((n/N^2)*100)
    end 

    return bandstructure
end 

function plot_bands_G_to_K(BZ,bandstructure,axes=gca())
    """
    This plots the bands along the direction between Gamma and K points in the Brillouin Zone
    This requires:
    - the bandstructure as a dictionary bandstructure[k] whose entries are the energies for that k vector
    - the Brillouin zone BZ as a matrix of k vectors 
    """
    num_bands = size(bandstructure[[0,0]])[1]

    GtoK = []
    bands_GtoK = [[] for i = 1:num_bands]
    E_matter_band_min = []

    E_matter_band = []

    for k in BZ 
        E_matter_min = 12 
        if (k[2]+k[1]==1) && k[1]>0.5
            push!(GtoK,k)
            for i in 1:num_bands
                if i%4 == 0 
                    if bandstructure[k][i] < E_matter_min
                        E_matter_min = bandstructure[k][i]
                    end 
                end 
                push!(bands_GtoK[i],bandstructure[k][i])
            end
            push!(E_matter_band_min,E_matter_min)
            push!(E_matter_band,get_exact_GS_matter_fermion_energy(k[1]*g1+k[2]*g2))
        end 
    end 
    kGtoK = collect((1:length(GtoK))*(g1[1]-g2[1])/(2*length(GtoK)))
    display(kGtoK)

    for i in 1:3
        axes.plot(kGtoK,bands_GtoK[i],color="black")
    end
    axes.plot(kGtoK,E_matter_band_min,color="b")
    axes.plot(kGtoK,E_matter_band,color="r")
end 

function shift_k_to_BZ(k_vec,N)
    shifted_k_vec = zeros(1,2)
    if k_vec[1] < 0
        shifted_k_vec[1] = ceil(k_vec[1]+1)%1
    else 
        shifted_k_vec[1] = ceil(k_vec[1])%1
    end 
    if k_vec[2] < 0
        shifted_k_vec[2] = ceil(k_vec[2]+1)%1
    else
        shifted_k_vec[2] = ceil(k_vec[2])%1
    end 

    return shifted_k_vec
end 

function plot_bands_G_to_K_to_M_to_G(BZ,bandstructure,axes=gca())

    num_bands = size(bandstructure[[0,0]])[1]
    GtoKtoMtoG = []
    GtoM = []
    bands_GtoKtoMtoG = [[] for i = 1:num_bands]
    bands_GtoM = [[] for i = 1:num_bands]

    E_matter_band_min_GtoKtoMtoG = []
    E_matter_band_min_GtoM = []

    E_matter_band_GtoKtoMtoG = []
    E_matter_band_GtoM = []


    K_index=0
    for k in BZ 
        E_matter_min = 12
        if (k[2] + k[1]== 1) && k[1] >=0.5
            push!(GtoKtoMtoG,k)
            for i in 1:num_bands
                push!(bands_GtoKtoMtoG[i],bandstructure[k][i])
                if i%4 == 0 
                    if bandstructure[k][i] < E_matter_min
                        E_matter_min = bandstructure[k][i]
                    end 
                end 
            end
            if k[1] < 2/3 && K_index ==0
                K_index = length(GtoKtoMtoG)
            end

            push!(E_matter_band_min_GtoKtoMtoG,E_matter_min)
            push!(E_matter_band_GtoKtoMtoG,get_exact_GS_matter_fermion_energy(k[1]*g1+k[2]*g2))
        elseif (k[1]==0) && k[2] <0.5
            push!(GtoM,k)
            for i in 1:num_bands
                push!(bands_GtoM[i],bandstructure[k][i])
                if i%4 == 0 
                    if bandstructure[k][i] < E_matter_min
                        E_matter_min = bandstructure[k][i]
                    end 
                end 
            end

            push!(E_matter_band_min_GtoM,E_matter_min)
            push!(E_matter_band_GtoM,get_exact_GS_matter_fermion_energy(k[1]*g1+k[2]*g2))
        end
    end 

    kGtoKtoMtoG = collect((1:length(GtoKtoMtoG))*(g1[1]-g2[1])/(2*length(GtoKtoMtoG)))
    kGtoM = collect((1:length(GtoM))*(g1[2]+g2[2])/(2*length(GtoM))) .+ ones(length(GtoM),1)*kGtoKtoMtoG[end]

    M_index = length(GtoKtoMtoG)

    append!(kGtoKtoMtoG,(kGtoM))
    for i =1:num_bands
        append!(bands_GtoKtoMtoG[i],reverse(bands_GtoM[i]))
    end 
    append!(E_matter_band_min_GtoKtoMtoG,reverse(E_matter_band_min_GtoM))
    append!(E_matter_band_GtoKtoMtoG,reverse(E_matter_band_GtoM))
    

    E_min= 0
    for i in 1:12000
        if E_min > minimum(bands_GtoKtoMtoG[i]) 
           E_min = minimum(bands_GtoKtoMtoG[i])
        end
        axes.plot(kGtoKtoMtoG,bands_GtoKtoMtoG[i],color="black")
    end

    E_max = 6

    axes.plot(kGtoKtoMtoG,E_matter_band_min_GtoKtoMtoG,color="b")
    axes.plot(kGtoKtoMtoG,E_matter_band_GtoKtoMtoG,color="r")

    display(K_index)
    display(M_index)
    
    axes.set_xticks([kGtoKtoMtoG[1],kGtoKtoMtoG[K_index],kGtoKtoMtoG[M_index],kGtoKtoMtoG[length(kGtoKtoMtoG)]])
    axes.set_xticklabels(["\$\\Gamma\$","K","M","\$\\Gamma\$"])
    axes.set_ylabel("Energy")
    axes.vlines([kGtoKtoMtoG[1],kGtoKtoMtoG[K_index],kGtoKtoMtoG[M_index],kGtoKtoMtoG[length(kGtoKtoMtoG)]],(1.1*E_min),1.1*E_max,linestyle="dashed")
end