
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
rz = (a1 + a2)/3
ry = (a1 - 2a2)/3
rx = (a2 - 2a1)/3

nz = [1,1]./3
ny = [1,-2]./3
nx = [-2,1]./3

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

g1,g2 = dual(a1,a2)

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
    M_flipped[C_A_index,C_B_index] = -M[C_A_index,C_B_index]
    #M[C_A_index,C_B_index] = 1
    return M_flipped
end 

function convert_n1_n2_to_site_index(n1n2vec,BCs)
    """
    given a lattice vector in the form [n1,n2] this function converts it to the corresponding site index, accounting for the periodic boundary conditions
    """
    n1 = n1n2vec[1]
    n2 = n1n2vec[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    # First translate n1,n2 back into the cell
    twist_num = floor(n2/L2)*m
    n2_prime = (n2+100*L2)%L2 # The + 100L2 term is to try and ensure n2_prime is positive  
    n1_prime = (n1 - twist_num + 100*L1)%L1

    site_index = Int(1+ L1*n2_prime + n1_prime)

    return site_index 
end 

function convert_bond_to_site_indicies(BCs,bond_site,bond_flavour)
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

    return C_A_index, C_B_index
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
    return M 
end 

function get_M_for_max_seperated_visons_along_a2_boundary(BCs)
    """
    Creates a chain of flipped z and y links across half of the a2 boundary, creating 2 visons at the ends of the chain
    The chain ends with flipped z links  
    """
    M = get_M0(BCs) 

    L1 = BCs[1]
    L2 = BCs[2]

    for j = Int(L1*(floor(L2/2))):-L1:1
        M[j+1,j+1] = -1 # flips the z link on the a2 boundary 
        M[j+1,j-L1+1] = -1 #flips the y link on the a2 boundary 
    end 
    M[1,1] = -1 # flips the z link at the origin 

    return M 
end 

function get_M_for_L_seperated_visons_along_a1_boundary(BCs,L_sep)
    M = get_M0(BCs) 

    L1 = BCs[1]
    L2 = BCs[2]

    B_flip = Int.(Matrix(I,L1,L1))
    B_flip[1:L_sep,1:L_sep] = -Int.(Matrix(I,L_sep,L_sep))

    M[(L1+1):2*L1,1:L1] = B_flip
    return M 
end 

function flip_bond_on_one_layer(M,BCs,bond_site,bond_flavour)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    f_A_index, f_B_index = convert_bond_to_site_indicies(BCs,bond_site,bond_flavour)

    M_flipped=zeros(L1*L2,L1*L2)
    A = zeros(L1*L2,L1*L2)
    M_flipped[:,:] = M
    M_flipped[f_A_index,f_B_index] = 0
    A[f_A_index,f_B_index] = 1

    return M_flipped, A 
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

    display("The determinant of U is...")
    display(det(U))
    display("The determinant of V is...")
    display(det(V))

    return D
end 

function get_K0(BCs)
    """
    Finds the on-diagonal matrix added by the extension term to the Kitaev model 
    Currently does not work for m non-zero
    """
    L1 = BCs[1]
    L2 = BCs[2]
    N = L1*L2

    P1 = zeros(L1,L1)
    P1[2:L1,1:L1-1] = Matrix(I,L1-1,L1-1)
    P1[1,L1] = 1

    P2 = zeros(L2,L2)
    P2[2:L2,1:L2-1] = Matrix(I,L2-1,L2-1)
    P2[1,L2] = 1

    display(P1)
    Mz = kron(I(L1),I(L2))
    Mx = kron(I(L2),P1)
    My = kron(P2,I(L1))

    A = Mx*My'+My*Mz'+Mz*Mx'

    K0 = A-A'

    return K0
end 

function get_Mx_My_Mz(M,BCs)
    L1 = BCs[1]
    L2 = BCs[2]
    N = L1*L2

    Mz = I(N).*M

    M_prime = M-Mz

    Mx = zeros(N,N)
    for n2 = 0:(L2-1)
        Mx[1+n2*L1:(n2+1)*L1,1+n2*L1:(n2+1)*L1] = M_prime[1+n2*L1:(n2+1)*L1,1+n2*L1:(n2+1)*L1]
    end 

    My = M - Mx - Mz

    return Mx, My, Mz
end 

function get_KA_KB(M,BCs,κ)
    Mx ,My ,Mz = get_Mx_My_Mz(M,BCs)

    KA = -κ*(Mx*My'-My*Mx'+My*Mz'-Mz*My'+Mz*Mx'-Mx*Mz')
    KB = -κ*(Mx'*My-My'*Mx+My'*Mz-Mz'*My+Mz'*Mx-Mx'*Mz)

    return KA , KB
end 

function get_extended_H_tilde(M,BCs,κ)
    KA, KB = get_KA_KB(M,BCs,κ)

    svd_M = svd(M)
    U = svd_M.U
    V = svd_M.V
    E = svd_M.S 

    KA_tilde = U'*KA*U
    KB_tilde = V'*KB*V

    H = [KA_tilde diagm(E) ; -diagm(E) KB_tilde]

    return 0.5*(H-H')
end 

function get_extended_H(M,BCs,κ)
    KA, KB = get_KA_KB(M,BCs,κ)

    H = [KA M ; -M' KB]

    return 0.5*(H-H')
end 

function get_TA_TB(M,BCs,κ)
    N = BCs[1]*BCs[2]
    H = get_extended_H(M,BCs,κ)

    T = im*eigvecs(im*H)
    T = [conj.(T[:,1:N]) T[:,1:N]]
    TA = T[1:N,1:N]
    TB = -im*(T[(N+1):2*N,1:N])

    return TA , TB
end 

function find_UA_from_TA(TA)
    N = size(TA)[1]
    N_2 = Int(floor(N/2))
    A = Hermitian(TA*TA')
    QA = im*eigvecs(A)
    #QA = [conj.(QA[:,1:N_2]) QA[:,1:N_2]]
    UA = (1/sqrt(2))*QA*[I(N_2) im*I(N_2) ; I(N_2) -im*I(N_2)]
    return UA
end 


function get_T_bar(M,BCs,κ)
    N = BCs[1]*BCs[2]
    H = get_extended_H(M,BCs,κ)

    T = im*eigvecs(im*H)
    T = [conj.(T[:,1:N]) T[:,1:N]]
    TA = T[1:N,1:N]
    TB = -im*conj.(T[(N+1):2*N,1:N])

    #T_bar = sqrt(2)*[real.(TA) imag.(TA) ; -imag.(TB) real.(TB)]
    T_bar = (1/sqrt(2))*T*[I(N) im*I(N) ; I(N) -im*I(N)]

    return real.(T_bar)
end 

function get_RA_IA(T_bar,BCs)
    N = BCs[1]*BCs[2]

    RA = T_bar[1:N,1:N]
    IA = T_bar[1:N,(N+1):2*N]
    RB = T_bar[(N+1):2*N,(N+1):2*N]
    IB = -T_bar[(N+1):2*N,1:N]
    return RA , RB, IA ,IB 
end 

function get_link_indices(BCs)
    N = BCs[1]*BCs[2]
    A_sites = [[n1,n2] for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    x_linked_B_sites = [r+nx-nz for r in A_sites]
    y_linked_B_sites = [r+ny-nz for r in A_sites]

    x_link_indices = [[j,convert_n1_n2_to_site_index(x_linked_B_sites[j],BCs)] for j = 1:N]
    y_link_indices = [[j,convert_n1_n2_to_site_index(y_linked_B_sites[j],BCs)] for j = 1:N]
    z_link_indices = [[j,j] for j=1:N]

    return x_link_indices, y_link_indices, z_link_indices
end 

function calculate_link_energies_EXT(M,BCs,κ,link_indices)
    T_bar = get_T_bar(M,BCs,κ)
    RA , RB ,IA , IB = get_RA_IA(T_bar,BCs)
    link_energy_matrix =  M.*(RA*RB'+IA*IB')
 
    link_energies = [link_energy_matrix[link_index[1],link_index[2]] for link_index in link_indices]
 
    return link_energies
end

function plot_bond_energies_2D_repeated_cell_EXT(M,BCs,κ)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    A_sites = [n1*a1+n2*a2 for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    shift_vectors = [-L2*a2-m*a1-L1*a1,-L2*a2-m*a1,-L2*a2-m*a1+L1*a1,-L1*a1,[0,0],L1*a1,L2*a2+m*a1-L1*a1,L2*a2+m*a1,L2*a2+m*a1+L1*a1]

    z_links = [(r,r+rz) for r in A_sites]
    x_links = [(r,r+rx) for r in A_sites]
    y_links = [(r,r+ry) for r in A_sites]

    x_link_indices, y_link_indices, z_link_indices = get_link_indices(BCs)

    link_energy_of_fluxless_system = 0.5250914141631111

    x_link_energies = calculate_link_energies_EXT(M,BCs,κ,x_link_indices) .- link_energy_of_fluxless_system
    y_link_energies = calculate_link_energies_EXT(M,BCs,κ,y_link_indices) .- link_energy_of_fluxless_system
    z_link_energies = calculate_link_energies_EXT(M,BCs,κ,z_link_indices) .- link_energy_of_fluxless_system

    cmap = get_cmap("seismic") # Should choose a diverging colour map so that 0.5 maps to white 

    max_energy = maximum(maximum.([abs.(x_link_energies),abs.(y_link_energies),abs.(z_link_energies)]))
    display(max_energy)

    xcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in x_link_energies]
    ycolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in y_link_energies]
    zcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in z_link_energies]

    ax=gca()
    for shift_vector in shift_vectors
        z_links = [(r+shift_vector,r+rz+shift_vector) for r in A_sites]
        x_links = [(r+shift_vector,r+rx+shift_vector) for r in A_sites]
        y_links = [(r+shift_vector,r+ry+shift_vector) for r in A_sites]
        
        zlinecollection = matplotlib.collections.LineCollection(z_links,colors = zcolors,linewidths=2)
        xlinecollection = matplotlib.collections.LineCollection(x_links,colors = xcolors,linewidths=2)
        ylinecollection = matplotlib.collections.LineCollection(y_links,colors = ycolors,linewidths=2)

        ax.add_collection(zlinecollection)
        ax.add_collection(xlinecollection)
        ax.add_collection(ylinecollection)
    end 
    ax.autoscale()
end 

function animate_bond_energies_increasing_magnetic_field(M,BCs,κ_max)
    for κ = LinRange(0,κ_max,10)
        plot_bond_energies_2D_repeated_cell_EXT(M,BCs,κ)
        gca().set_xlim([-10,10])
        gca().set_ylim([-10,10])
        sleep(0.1)
    end 
end 

function plot_Majorana_distribution_EXT(BCs,c_distribution,max=maximum(c_distribution),sublattice="A", ax2 = gca(),repeat=false)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    #cmap = get_cmap("PuOr")
    cmap = get_cmap("seismic")


    #=
    fig = gcf()
    if size(fig.axes)[1] > 1
        (fig.axes[2]).clear()
        display("Clearing current Majorana distribution...")
        ax1 = fig.axes[1]
        ax2 = ax1.twinx()
        y_min,y_max = ax1.get_ylim()
        ax2.set_ylim([y_min,y_max])
    else 
        ax2 = gca()
    end 
    =#

    shift_vectors = [-L2*a2-m*a1-L1*a1,-L2*a2-m*a1,-L2*a2-m*a1+L1*a1,-L1*a1,[0,0],L1*a1,L2*a2+m*a1-L1*a1,L2*a2+m*a1,L2*a2+m*a1+L1*a1]
    if repeat == false
        shift_vectors = [[0,0]]
    end 

    if sublattice == "B"
        site_shift = rz[2]
    else
        site_shift = 0
    end 

    for shift_vec in shift_vectors
        ax2.scatter(A_site_x_coords.+shift_vec[1],A_site_y_coords.+shift_vec[2].+site_shift,color = cmap.((c_distribution./max).+0.5),s=6,zorder=2)
    end 
end

function plot_eigenstate_Majorana_wavefunction_EXT(M,BCs,κ,ex_id)
    T_bar = get_T_bar(M,BCs,κ)
    RA , RB , IA ,IB = get_RA_IA(T_bar,BCs)

    fig , axes =  subplots(1,2)
    # Cα
    plot_Majorana_distribution_EXT(BCs,RA[:,end-ex_id+1],maximum(RA[:,end-ex_id+1]),"A",axes[1],true)
    plot_Majorana_distribution_EXT(BCs,-IB[:,end-ex_id+1],maximum(RA[:,end-ex_id+1]),"B",axes[1],true)
    #Cβ
    plot_Majorana_distribution_EXT(BCs,IA[:,end-ex_id+1],maximum(RB[:,end-ex_id+1]),"A",axes[2],true)
    plot_Majorana_distribution_EXT(BCs,RB[:,end-ex_id+1],maximum(RB[:,end-ex_id+1]),"B",axes[2],true)
end 

BCs = [20,20,0]
N=BCs[1]*BCs[2]

M0 = get_M0(BCs)
Mv = flip_bond_variable(M0,BCs,[1,1],"y")
M2v = flip_bond_variable(Mv,BCs,[2,1],"x")
M_max = get_M_for_max_seperated_visons_along_a2_boundary(BCs)


κ = 0.2
T_bar_v = get_T_bar(Mv,BCs,κ)
T_bar_0 = get_T_bar(M0,BCs,κ)

T_bar = T_bar_0'*T_bar_v

RA0,RB0,IA0,IB0 = get_RA_IA(T_bar_0,BCs)
RA,RB,IA,IB = get_RA_IA(T_bar,BCs)

U = RA + im*IB
V = RB + im*IA 

TA0 = (RA0+im*IA0)/sqrt(2)
TB0 = (RB0+im*IB0)/sqrt(2)

TA = (RA+im*IA)/sqrt(2)
TB = (RB+im*IB)/sqrt(2)

F = real.(V'*U)
z = eigvecs(F)[:,1]
O = angle.(eigvals(F))[1]
b_bar = sqrt(2)*imag.(z)
a_bar = sqrt(2)*real.(z)

a_A = transpose(a_bar'*(RA'*RA0'-IB'*IA0'))
a_B = transpose(a_bar'*(RA'*IB0'+IB'*RB0'))

b_A = transpose(b_bar'*(RA'*RA0'-IB'*IA0'))
b_B = transpose(b_bar'*(RA'*IB0'+IB'*RB0'))


X = 0.5*(U'+V')
Y = 0.5*transpose(U-V)

Z = inv(X)*Y
Z = round.(Z,digits=15)

function plot_Z_eigvals_increasing_kappa()
    for κ = LinRange(0,0.1,16)
        T_bar_v = get_T_bar(M2v,BCs,κ)
        T_bar_0 = get_T_bar(Mv,BCs,κ)

        T_bar = T_bar_0'*T_bar_v

        RA0,RB0,IA0,IB0 = get_RA_IA(T_bar_0,BCs)
        RA,RB,IA,IB = get_RA_IA(T_bar,BCs)

        U = RA + im*IB
        V = RB + im*IA 

        X = 0.5*(U'+V')
        Y = 0.5*transpose(U-V)

        Z = inv(X)*Y

        SZ = svd(Z).S
        scatter(κ, SZ[1])
        scatter(κ, SZ[3])
        sleep(0.1)
    end 
end
UZ ,VZ = get_U_and_V(Z)
FZ = VZ'*conj.(UZ)
FZ = FZ.*round.(abs.(FZ),digits=0)
Phi = kron(I(Int(floor(N/2))),[0 -1 ; 1 0])*FZ
Phi = diag(angle.(Phi))
Phi = diagm(exp.(im*Phi))
 
SZ = diagm(svd(Z).S)

u1 = UZ[:,1]
u2 = UZ[:,2]

UZ_prime = UZ*(Phi.^(0.5))
u1_prime = UZ_prime[1,:]
u2_prime = UZ_prime[2,:]

A1 = TA0*u1
B1 = TB0*u1
#=
svd_U = svd(U)
Uu = svd_U.U
Vu = svd_U.V
Su = diagm(svd_U.S)

svd_V = svd(V) 
Sv = diagm(reverse(svd_V.S))
Vv = (inv(Sv)*Uu'*V)' 

F = real.(Vu*Vv')

Rv = real.(Vv)
Iv = imag.(Vv)
P = Sv+(Su-Sv)*Rv'*Rv

Ru = real.(Vu)
Iu = imag.(Vu)
P = Sv+(Su-Sv)*Ru'*Ru

Vr = svd(Ru).V
Ur = svd(Ru).U 
Ui = P*Iu*Vr*diagm(reverse(svd(Iu).S))

Q = eigvecs(F)
Θ = diagm(eigvals(F))

Y_prime = 0.25*(P+P')*transpose(Vu)*Q*(I(N)-Θ).+(P-P')*Vu'*Q*(I(N)+Θ)
X_prime = 0.25*(P+P')*transpose(Vu)*Q*(I(N)+Θ).+(P-P')*Vu'*Q*(I(N)-Θ)

YP = 0.5*(P-P')
XP = 0.5*(P+P')
Xt = 0.5*(I(N)+Θ)
Yt = 0.5*(I(N)-Θ)

g = zeros(N,N)
for i = 1:N
    g[N+1-i,i] = 1
end 
=#

function plot_real_part_of_F21_for_L_seperated_visons_Vs_BCs_EXT(L_max,L_sep,κ=0.01)
    """
    This calculates and plots real parts of eigenvalues of F21 for visons seperated by L_sep parallel "z" bond flips
    For lattice sizes with lengths multiples of 3 the determinant det(F21) is fixed to be the opposite sign as for other lattice sizes where it has a well defined value. 
    """
    L_min = 2*L_sep+1
    if L_min%3 != 0 
        BCs = [L_min,L_min,0]
    else
        BCs = [L_min+1,L_min+1,0]
    end 
    M0 = get_M0(BCs)
    Mv = get_M0(BCs)
    # Creates an M matrix for visons seperated by L_sep parallel bond flips 
    for j = 0:(L_sep-1)
        Mv = flip_bond_variable(Mv,BCs,[L_sep+j,L_sep-j],"z")   
    end 

    T_bar_v = get_T_bar(Mv,BCs,0.01)
    T_bar_0 = get_T_bar(M0,BCs,0.01)

    T_bar = T_bar_0'*T_bar_v

    RA,RB,IA,IB = get_RA_IA(T_bar,BCs)

    U = RA + im*IB
    V = RB + im*IA 

    F = U'*V

    det_F = Int(round(det(F)))

    for L = L_min:L_max
        BCs = [L,L,0]
        M0 = get_M0(BCs)
        Mv = get_M0(BCs)
        for j = 0:(L_sep-1)
            Mv = flip_bond_variable(Mv,BCs,[L_sep+j,L_sep-j],"z")
        end 

        T_bar_v = get_T_bar(Mv,BCs,κ)
        T_bar_0 = get_T_bar(M0,BCs,κ)

        T_bar = T_bar_0'*T_bar_v

        RA,RB,IA,IB = get_RA_IA(T_bar,BCs)

        U = RA + im*IB
        V = RB + im*IA 

        F = real.(U'*V)

        #=
        if L%3 == 0 && Int(round(det(F))) != -(det_F)
            U0[:,end] = -U0[:,end]
            U = Uv'*U0
            F = U'*V
        end 
        =#

        cos_θ = real.(eigvals(F))[1:5]
        display(cos_θ)
        scatter(L*ones(5),cos_θ,color="b")
        sleep(0.01)
    end 
    title("Real part of F21 eigenvalues, $L_sep bonds flipped, k=$κ K")
    xlabel("Linear lattice size L")
    ylabel("cos(θ)")
end 

#=

UA ,VA = get_U_and_V(RA)
VB ,UB = get_U_and_V(RB)

T1 = [I(N) zeros(N,N) ; zeros(N,N) UA*VB']*T_bar*[I(N) zeros(N,N) ; zeros(N,N) UB*VA']
T2 = [RA IA*UB*VA' ; - IA*UB*VA' RA]

I_prime = UA'*IA*UB

R = diagm(round.(svd(RA).S,digits=3))

function svd_non_zero_block(RA,IA,BCs)
    N=BCs[1]*BCs[2]
    I_prime = UA'*IA*UB
    R = diagm(round.(svd(RA).S,digits=3))

    block_index = Int(sum(R.>1e-10))

    I_block = I_prime[block_index+1:end,block_index+1:end]

    U_block ,V_block = get_U_and_V(I_block)

    UI = zeros(N,N)
    UI[block_index+1:end,block_index+1:end] = U_block
    VI = zeros(N,N)
    VI[block_index+1:end,block_index+1:end] = V_block
    UI[1:(block_index),1:(block_index)] = I(block_index)
    VI[1:(block_index),1:(block_index)] = I(block_index)

    return UI,VI
end 

UI ,VI = svd_non_zero_block(RA,IA,BCs)

=#

display("")