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

# This sections adds functions for plotting the real space lattice 
function plot_real_space_lattice(BCs)
    lattice_sites = zeros(BCs[1]*BCs[2],2)

    y_extent = 4 #  Int(round((L1*a1+L2*a2)[2]*(1/(2*sqrt(3)))))+3
    x_extent = 4 # Int(round((L1*a1-L2*a2)[1]/2))+2
    display(x_extent)

    site_index=1
    for n2 = 0:(BCs[2]-1)
        for n1 = 0:(BCs[1]-1)
            r = n1*a1+n2*a2 
            lattice_sites[site_index,:] = r
            site_index +=1
        end 
    end 

    hexbin(lattice_sites[:,1],lattice_sites[:,2].-1*nz[2],gridsize=(x_extent,y_extent),edgecolors="w")
    scatter(lattice_sites[:,1],lattice_sites[:,2])
end 

function plot_links(BCs)

    A_sites = [n1*a1+n2*a2 for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    z_links = [(r,r+rz) for r in A_sites]
    x_links = [(r,r+rx) for r in A_sites]
    y_links = [(r,r+ry) for r in A_sites]

    zlinecollection = matplotlib.collections.LineCollection(z_links)
    xlinecollection = matplotlib.collections.LineCollection(x_links)
    ylinecollection = matplotlib.collections.LineCollection(y_links)

    fig ,ax = subplots()
    ax.add_collection(zlinecollection)
    ax.add_collection(xlinecollection)
    ax.add_collection(ylinecollection)
    ax.autoscale()

    return z_links
end 

#This section looks at energetics of adding visons to the system

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

function plot_GS_energy_max_seperated_visons_vs_system_size(L_Max,m)
    """
    Uses an L by L lattice 
    """
    for L = 5:L_Max
        BCs = [L,L,m]
        M = get_M_for_single_visons(BCs)
        M0 = get_M0(BCs)
        M_pair = flip_bond_variable(M0,BCs,[0,0],"z")

        GS_energy_per_site_w_visons = -sum(svd(M).S)/L^2
        GS_energy_per_site_fluxless = -sum(svd(M0).S)/L^2
        GS_energy_per_site_vison_pair = -sum(svd(M_pair).S)/L^2

        scatter(1/L,GS_energy_per_site_w_visons,color="b")
        scatter(1/L,GS_energy_per_site_fluxless,color="r")
        scatter(1/L,GS_energy_per_site_vison_pair,color="g")
    end 
end 

function plot_energy_shift_due_to_vison_pair(BCs)
    M0 = get_M0(BCs)
    M = flip_bond_variable(M0,BCs,[0,0],"z")

    E0 = svd(M0).S
    E = svd(M).S

    plot(E0,E-E0)
end 

# This section adds functions for plotting the bond energy distribution around visons 

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

function calculate_F(M)
    U,V = get_U_and_V(M)
    F = U*V'

    return F
end 

function calculate_link_energies(F,M,link_indices)
   link_energy_matrix =  M.*F

   link_energies = [link_energy_matrix[link_index[1],link_index[2]] for link_index in link_indices]

   return link_energies
end

function plot_bond_energies_2D(M,BCs)

    F = calculate_F(M)

    A_sites = [n1*a1+n2*a2 for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    z_links = [(r,r+rz) for r in A_sites]
    x_links = [(r,r+rx) for r in A_sites]
    y_links = [(r,r+ry) for r in A_sites]

    x_link_indices, y_link_indices, z_link_indices = get_link_indices(BCs)

    link_energy_of_fluxless_system = 0.5250914141631111

    x_link_energies = calculate_link_energies(F,M,x_link_indices) .- link_energy_of_fluxless_system
    y_link_energies = calculate_link_energies(F,M,y_link_indices) .- link_energy_of_fluxless_system
    z_link_energies = calculate_link_energies(F,M,z_link_indices) .- link_energy_of_fluxless_system

    cmap = get_cmap("seismic") # Should choose a diverging colour map so that 0.5 maps to white 

    max_energy = maximum(maximum.([abs.(x_link_energies),abs.(y_link_energies),abs.(z_link_energies)]))
    display(max_energy)

    xcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in x_link_energies]
    ycolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in y_link_energies]
    zcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in z_link_energies]

    zlinecollection = matplotlib.collections.LineCollection(z_links,colors = zcolors,linewidths=2.5)
    xlinecollection = matplotlib.collections.LineCollection(x_links,colors = xcolors,linewidths=2.5)
    ylinecollection = matplotlib.collections.LineCollection(y_links,colors = ycolors,linewidths=2.5)

    fig ,ax = subplots()
    ax.add_collection(zlinecollection)
    ax.add_collection(xlinecollection)
    ax.add_collection(ylinecollection)
    ax.autoscale()

end 

function plot_bond_energies_2D_repeated_cell(M,BCs)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]
    
    F = calculate_F(M)

    A_sites = [n1*a1+n2*a2 for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    shift_vectors = [-L2*a2-m*a1-L1*a1,-L2*a2-m*a1,-L2*a2-m*a1+L1*a1,-L1*a1,[0,0],L1*a1,L2*a2+m*a1-L1*a1,L2*a2+m*a1,L2*a2+m*a1+L1*a1]

    z_links = [(r,r+rz) for r in A_sites]
    x_links = [(r,r+rx) for r in A_sites]
    y_links = [(r,r+ry) for r in A_sites]

    x_link_indices, y_link_indices, z_link_indices = get_link_indices(BCs)

    link_energy_of_fluxless_system = 0.5250914141631111

    x_link_energies = calculate_link_energies(F,M,x_link_indices) .- link_energy_of_fluxless_system
    y_link_energies = calculate_link_energies(F,M,y_link_indices) .- link_energy_of_fluxless_system
    z_link_energies = calculate_link_energies(F,M,z_link_indices) .- link_energy_of_fluxless_system

    cmap = get_cmap("seismic") # Should choose a diverging colour map so that 0.5 maps to white 

    max_energy = maximum(maximum.([abs.(x_link_energies),abs.(y_link_energies),abs.(z_link_energies)]))
    display(max_energy)

    xcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in x_link_energies]
    ycolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in y_link_energies]
    zcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in z_link_energies]

    fig ,ax = subplots()
    for shift_vector in shift_vectors
        z_links = [(r+shift_vector,r+rz+shift_vector) for r in A_sites]
        x_links = [(r+shift_vector,r+rx+shift_vector) for r in A_sites]
        y_links = [(r+shift_vector,r+ry+shift_vector) for r in A_sites]
        
        zlinecollection = matplotlib.collections.LineCollection(z_links,colors = zcolors,linewidths=2.5)
        xlinecollection = matplotlib.collections.LineCollection(x_links,colors = xcolors,linewidths=2.5)
        ylinecollection = matplotlib.collections.LineCollection(y_links,colors = ycolors,linewidths=2.5)

        ax.add_collection(zlinecollection)
        ax.add_collection(xlinecollection)
        ax.add_collection(ylinecollection)
    end 
    ax.autoscale()
end 

function plot_low_energy_Majorana_distribution(M,BCs,excited_state_num=0)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]

    SVD_M = svd(M)
    U = SVD_M.U
    V = (SVD_M.V)
    E = SVD_M.S

    State_energy = E[N-excited_state_num]
    display("Energy is $State_energy")

    cA_distribution = U[:,(N-excited_state_num)]
    cB_distribution = V[:,(N-excited_state_num)]

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    cmap = get_cmap("PuOr")
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
    ax2.scatter(A_site_x_coords,A_site_y_coords,color = cmap.((cA_distribution./maximum(cA_distribution)).+0.5),zorder=2)
    #ax2.scatter(A_site_x_coords,A_site_y_coords.+rz[2],color = cmap.((cB_distribution./maximum(cA_distribution)).+0.5),zorder=2)
    ax2.scatter(0,0,alpha=0,zorder=4)
end

function plot_low_energy_Majorana_distribution_repeated_cells(M,BCs,excited_state_num=0)
    N = BCs[1]*BCs[2]
    
    SVD_M = svd(M)
    U = SVD_M.U
    V = (SVD_M.V)
    E = SVD_M.S

    State_energy = E[N-excited_state_num]
    display("Energy is $State_energy")

    cA_distribution = U[:,(N-excited_state_num)]
    cB_distribution = V[:,(N-excited_state_num)]

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    cmap = get_cmap("PuOr")
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

    shift_vectors = [-L2*a2-m*a1-L1*a1,-L2*a2-m*a1,-L2*a2-m*a1+L1*a1,-L1*a1,[0,0],L1*a1,L2*a2+m*a1-L1*a1,L2*a2+m*a1,L2*a2+m*a1+L1*a1]

    for shift_vec in shift_vectors
        ax2.scatter(A_site_x_coords.+shift_vec[1],A_site_y_coords.+shift_vec[2],color = cmap.((cA_distribution./maximum(cA_distribution)).+0.5),zorder=2,s=1.9,linewidths=0)
        ax2.scatter(A_site_x_coords.+shift_vec[1],A_site_y_coords.+rz[2].+shift_vec[2],color = cmap.((cB_distribution./maximum(cA_distribution)).+0.5),zorder=2,s=1.9,linewidths=0)
    end 
    ax2.scatter(0,0,alpha=0,zorder=4)
end

function plot_F_at_site_n1_n2(M,BCs,r_n1_n2)
    N = BCs[1]*BCs[2]
    
    SVD_M = svd(M)
    U = SVD_M.U
    V = (SVD_M.Vt)'
    F = U*V'
    display(F)

    site_r_index = convert_n1_n2_to_site_index(r_n1_n2,BCs)
    display(site_r_index)

    F_r_R = F[:,site_r_index]

    display(F_r_R)
    F_max = maximum(abs.(F_r_R))
    display(F_max)

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    cmap = get_cmap("seismic")
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

    ax2.scatter(A_site_x_coords,A_site_y_coords.+rz[2],color = cmap.((F_r_R./F_max).+0.5),zorder=2)
    ax2.scatter(0,0,alpha=0,zorder=4)
end

function plot_delta_F_at_site_n1_n2(M,BCs,r_n1_n2)
    N = BCs[1]*BCs[2]
    
    SVD_M = svd(M)
    U = SVD_M.U
    V = (SVD_M.Vt)'
    F = U*V'
   
    F0 = calculate_F(M0)

    delta_F = F-F0

    site_r_index = convert_n1_n2_to_site_index(r_n1_n2,BCs)
   

    delta_F_r_R = delta_F[:,site_r_index]

    delta_F_max = maximum(abs.(delta_F_r_R))
    display(delta_F_max)

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    cmap = get_cmap("seismic")
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

    ax2.scatter(A_site_x_coords,A_site_y_coords.+rz[2],color = cmap.((delta_F_r_R./delta_F_max).+0.5),zorder=2)
    ax2.scatter(0,0,alpha=0,zorder=4)
end

function plot_greater_Greens_function_for_f_fermions(M,BCs,r1_n1_n2=[0,0],r2_n1_n2=[0,0])
    SVD_M = svd(M)
    U=SVD_M.U
    V=SVD_M.V
    E=reverse(SVD_M.S)

    X = 0.5*(U+V)

    Num_bins = 25

    w = LinRange(0,3,Num_bins)

    G_gtr_w = zeros(Num_bins)

    binsize = 3/(Num_bins-1)

    site_index1 = convert_n1_n2_to_site_index(r1_n1_n2,BCs)
    site_index2 = convert_n1_n2_to_site_index(r2_n1_n2,BCs)

    j=1
    Energy = E[1]
    for (bin_index,E_bin) in enumerate(w)
        display(E_bin)
        while Energy < E_bin + binsize && j <= size(E)[1]
            Energy = E[j]
            G_gtr_w[bin_index] += X[site_index1,j]*X[site_index2,j]
            j +=1 
        end 
        display(G_gtr_w[bin_index])
    end 

    plot(w,G_gtr_w)
end

function plot_Greens_function_for_f_fermions(M,BCs,r_n1_n2=[0,0])
    SVD_M = svd(M)
    U=SVD_M.U
    V=SVD_M.V
    E=(SVD_M.S).*2

    X = 0.5*(U+V)

    Num_bins = 25
    eta = 10^(-6)

    w = reverse(LinRange(0,6,Num_bins))

    Im_G_w = zeros(Num_bins)
    Re_G_w = zeros(Num_bins)

    binsize = 6/(Num_bins-1)

    site_index = convert_n1_n2_to_site_index(r_n1_n2,BCs)

    onsite_density = 0

    j=1
    Energy = E[1]
    for (bin_index,E_bin) in enumerate(w)
        display(E_bin)
        while Energy > E_bin - binsize/2 && j <= size(E)[1]
            Energy = E[j]
            Im_G_w[bin_index] += pi*X[1,j]*X[site_index,j]/(binsize)
            onsite_density += X[1,j]*X[site_index,j]
            j +=1 
        end 
        #Re_G_w[bin_index] = (X*diagm((E.-E_bin)./(((E.-E_bin).^2).+eta^2))*X')[site_index,site_index]
    end 
    display(onsite_density)

    Im_G_w[1] = 2*Im_G_w[1]
    plot(w,Im_G_w)
    #plot(w,Re_G_w)
end

function plot_f_fermion_density(M,BCs)
    N = BCs[1]*BCs[2]
    
    SVD_M = svd(M)
    U = SVD_M.U
    V = (SVD_M.Vt)'
    E = SVD_M.S

    X=0.5*(U+V)
    Y=0.5*(U-V)

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    delta_f_density = zeros(N)
    f0_density = 0.762436
    for site_index  = 1:N
        delta_f_density[site_index] = (X*X')[site_index,site_index] - f0_density
    end 
    display(delta_f_density)

    cmap = get_cmap("seismic") # SHould choose a sequential colour map 
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
    ax2.scatter(A_site_x_coords,A_site_y_coords,color = cmap.((delta_f_density./maximum(delta_f_density)).+0.5),zorder=2)
    #ax2.scatter(A_site_x_coords,A_site_y_coords.+rz[2],color = cmap.((cB_distribution./maximum(cA_distribution)).+0.5),zorder=2)
    ax2.scatter(0,0,alpha=0,zorder=4)
end

# This section adds functions to calculate hopping parameters for the Bilayer model 

function calculate_interlayer_pair_hopping_for_min_seperated_visons(BCs)
    M0 = get_M0(BCs)
    M_1 = flip_bond_variable(M0,BCs,[0,0],"z")
    M_2 = flip_bond_variable(M0,BCs,[0,0],"y") # hops the vison by flipping 2 bonds, moving the vison at the origin in the a2 direction

    # matrices labelled 1 refer to the original link configuration (vison pair with z orientation at the origin)
    U1,V1 = get_U_and_V(M_1)
    # matrices labelled 2 refer to the hopped vison link configuration (vison pair with y orientation at the origin)
    U2,V2 = get_U_and_V(M_2)

    X21 = 0.5*(U2'*U1+V2'*V1)
    Y21 = 0.5*(U2'*U1-V2'*V1)
    

    #Z21 = inv(X21)*Y21
    hop_parameter = abs(det(X21))*(1+(U1*V1')[1,1])
    D1 = calculate_D(BCs,U1,V1,1)

    #display(det(U1))
    #display(det(V1))
    D2 = calculate_D(BCs,U2,V2,1)
    #display(det(U2))
    #display(det(V2))

    #display(det(X21))
    #display(hop_parameter)
    return hop_parameter, D1, D2
end 

function calculate_interlayer_pair_hopping_for_max_seperated_visons(BCs)
    M_max = get_M_for_max_seperated_visons_along_a2_boundary(BCs)
    M_hopped = flip_bond_variable(M_max,BCs,[0,0],"z")
    M_hopped = flip_bond_variable(M_hopped,BCs,[0,1],"y") # hops the vison by flipping 2 bonds, moving the vison at the origin in the a2 direction

    # matrices labelled 1 refer to the original link configuration (max seperation)
    U1,V1 = get_U_and_V(M_max)
    # matrices labelled 2 refer to the hopped vison link configuration 
    U2,V2 = get_U_and_V(M_hopped)

    X21 = 0.5*(U2'*U1+V2'*V1)
    Y21 = 0.5*(U2'*U1-V2'*V1)
    

    #Z21 = inv(X21)*Y21
    hop_parameter = det(X21)*(1+(U1*V1')[1,1])
    D1 = calculate_D(BCs,U1,V1,Int(floor(BCs[2]/2)))

    display(det(U1))
    display(det(V1))
    D2 = calculate_D(BCs,U2,V2,Int(floor(BCs[2]/2))-2)
    display(det(U2))
    display(det(V2))

    display(det(X21))
    #display(hop_parameter)
    return hop_parameter, D1, D2
end 

function plot_interlayer_pair_hopping_vs_system_size(N_max,twist,N_min=5)
    hopping_values = zeros(N_max-N_min+1)
    mark = "o"

    colours = ["blue","red","green","orange"]

    for (id,N) = enumerate(N_min:N_max)
        hopping_values[id], D1, D2  = calculate_interlayer_pair_hopping_for_max_seperated_visons([Int(ceil(N/2)),N,twist])
        if D1 == D2
            if D1 == 1 
                mark = "o"
            else
                mark= "^"
            end 
        else
            mark = "x"
        end 
        scatter(1/N,hopping_values[id],marker=mark,color = colours[twist+1])
    end 
end 

function plot_pair_hopping_vs_system_size(N_max,twist,N_min=5)
    hopping_values = zeros(N_max-N_min+1)
    mark = "o"

    colours = ["red","green","orange"]

    for (id,N) = enumerate(N_min:N_max)
        hopping_values[id], D1, D2  = calculate_interlayer_pair_hopping_for_min_seperated_visons([Int(ceil(N/2)),N,twist])
        if D1 == D2
            if D1 == 1 
                mark = "o"
            else
                mark= "^"
            end 
        else
            mark = "x"
        end 
        scatter(1/N,hopping_values[id],marker=mark,color = colours[twist+1])
    end 
end 

function calculate_parity_of_fluxless_ground_state(L1_max,L2_max,twist)
    for L1 = 3:L1_max
        for L2 = 3:L2_max
            M0 = get_M0([L1,L2,twist])
            Mv = flip_bond_variable(M0,[L1,L2,twist],[0,0],"z")
            #Mv = flip_bond_variable(Mv,[L1,L2,twist],[0,1],"y")
            U,V = get_U_and_V(M0)
            D = calculate_D(M0,U,V,0)
            if D == 1
                scatter(L1,L2,color="r")
            end 
        end 
    end 
end 

function plot_det_M0_vs_BCs(L1_max,L2_max,twist)
    for L1 = 3:L1_max
        for L2 = 3:L2_max
            M0 = get_M0([L1,L2,twist])
            det_M = det(M0)
            sgn_det = sign(det_M)
            if (svd(M0).S)[end] >0.1
                if sgn_det == 1
                    scatter(L1,L2,color="r")
                elseif sgn_det == -1
                    scatter(L1,L2,color="b")
                end 
            end 
        end 
    end 
end 

function plot_det_Mv_2_flips_vs_BCs(L1_max,L2_max,twist)
    for L1 = 3:L1_max
        for L2 = 3:L2_max
            M0 = get_M0([L1,L2,twist])
            Mv = flip_bond_variable(M0,[L1,L2,twist],[0,0],"x")
            Mv = flip_bond_variable(M0,[L1,L2,twist],[0,0],"y")
            det_M = det(Mv)
            sgn_det = sign(det_M)
            if (svd(Mv).S)[end] >0.02
                if sgn_det == 1
                    scatter(L1,L2,color="r")
                elseif sgn_det == -1
                    scatter(L1,L2,color="b")
                end 
            end 
        end 
    end 
end 

function plot_det_Mv_1_flip_vs_BCs(L1_max,L2_max,twist)
    for L1 = 3:L1_max
        for L2 = 3:L2_max
            M0 = get_M0([L1,L2,twist])
            Mv = flip_bond_variable(M0,[L1,L2,twist],[0,1],"z")
            det_M = det(Mv)
            sgn_det = sign(det_M)
            if (svd(Mv).S)[end] >0.02
                if sgn_det == 1
                    scatter(L1,L2,color="r")
                elseif sgn_det == -1
                    scatter(L1,L2,color="b")
                end 
            end 
        end 
    end 
end 

function get_A0(L1)
    A = zeros(L1,L1)
    for j = 1:L1-1
        A[j,j] = 1
        A[j+1,j] = 1
    end 
    A[L1,L1] = 1
    A[1,L1] = 1

    return A 
end 

function plot_change_in_det_M_max_vs_BCs(L1_max,L2_max,twist)
    for L1 = 3:L1_max
        for L2 = 3:L2_max
            M_max = get_M_for_max_seperated_visons_along_a2_boundary([L1,L2,twist])
            M_hop = flip_bond_variable(M_max,[L1,L2,twist],[0,0],"z")
            det_M = det(M_max)
            det_M_hop = det(M_hop)
            sgn_det = sign(det_M)*sign(det_M_hop)
            if (svd(M_max).S)[end] >0.01
                if sgn_det == 1
                    scatter(L1,L2,color="r")
                elseif sgn_det == -1
                    scatter(L1,L2,color="b")
                end 
            end 
        end 
    end 
end 

# This section uses the variational ansatz for the ground state with a pair of visons
function Delta_k(k)
    Delta = 1 + exp(im*dot(k,a1)) + exp(im*dot(k,a2))

    cos_phi = real(Delta)/abs(Delta)

    return abs(Delta) , cos_phi
end 
function expection_value_of_V(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    V = 0 
    for k in k_lattice 
        D_k ,cos_phi_k = Delta_k(k)
        V+=cos_phi_k
    end 
    return 2*V/(N)
end 

function self_consistent_equation(δ,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 
    for k in k_lattice
        Δ_k = 1 + exp(im*dot(k,a1)) + exp(im*dot(k,a2))
        for k_prime in k_lattice
            Δ_k_prime = 1 + exp(im*dot(k_prime,a1)) + exp(im*dot(k_prime,a2))
            Sum_term += (1-cos(angle(Δ_k)-angle(Δ_k_prime)))/(δ + 2*abs(Δ_k) + 2*abs(Δ_k_prime))
        end 
    end 
    return -(4/(N^2))*Sum_term
end 

function get_Z_kk_0(BCs,δ)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Z_kk = zeros(Complex{Float64},N,N)
    for (k_id,k) in enumerate(k_lattice)
        Δ_k = 1 + exp(im*dot(k,a1)) + exp(im*dot(k,a2))
        for (k_prime_id,k_prime) in enumerate(k_lattice)
            Δ_k_prime = 1 + exp(im*dot(k_prime,a1)) + exp(im*dot(k_prime,a2))
            Z_kk[k_id,k_prime_id] = (real(Δ_k)/abs(Δ_k)-real(Δ_k_prime)/abs(Δ_k_prime)+ im*(imag(Δ_k)/abs(Δ_k)-imag(Δ_k_prime)/abs(Δ_k_prime)))/(δ + 2*abs(Δ_k) + 2*abs(Δ_k_prime))
        end 
    end 

    return Z_kk/N
end 

function plot_Z_kk_in_BZ(BCs,δ,k_id)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]
    Z_kk = get_Z_kk_0(BCs,δ)
    k = k_lattice[k_id]
    kx_points = zeros(N)
    ky_points = zeros(N)
    z_points = zeros(N)

    for (k_prime_id,k_prime) in enumerate(k_lattice)
        kx_points[k_prime_id] = k_prime[1]
        ky_points[k_prime_id] = k_prime[2]
        z_points[k_prime_id] = abs(Z_kk[k_id,k_prime_id])
    end 

    display(z_points)
    scatter3D(kx_points,ky_points,z_points)
end 

function sum_of_Zkk_squared(δ,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 
    for k in k_lattice
        D_k ,cos_phi_k = Delta_k(k)
        for k_prime in k_lattice
            D_k_prime , cos_phi_k_prime = Delta_k(k_prime)
            Sum_term += ((cos_phi_k -cos_phi_k_prime)^2)/(delta + 2*D_k + 2*D_k_prime)^2
        end 
    end 
    return Sum_term/N^2
end 
        
function plot_self_consistent_eq(BCs)
    Num_points = 100 
    δ_points = LinRange(0,2,Num_points)
    eq_points = zeros(Num_points)
    for (id,δ) in enumerate(δ_points)
        eq_points[id] = self_consistent_equation(δ,BCs)
    end 
    plot(δ_points,eq_points)
    plot(δ_points,-δ_points)
end 

function find_Z_exact_for_vison_pair(BCs)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)
    X0, Y0 = 0.5(U0+V0) , 0.5*(U0-V0)

    Mv = flip_bond_variable(M0,BCs,[8,8],"z")
    Uv ,Vv = get_U_and_V(Mv)
    Xv, Yv = 0.5*(Uv+Vv) , 0.5*(Uv-Vv)

    U = Uv'*U0
    V = Vv'*V0
    X , Y = 0.5*(U+V) , 0.5*(U-V)

    display(det(X))
    Z = inv(X)*Y

    return 0.5*(Z-Z')
end 


# This section plots changes to the ground state in position space 

function plot_lowest_energy_pair_single_bond_flip(BCs)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)
    X0, Y0 = 0.5(U0+V0) , 0.5*(U0-V0)

    Mv = flip_bond_variable(M0,BCs,[8,8],"z")
    Uv ,Vv = get_U_and_V(Mv)
    Xv, Yv = 0.5*(Uv+Vv) , 0.5*(Uv-Vv)

    U = Uv'*U0
    V = Vv'*V0
    X , Y = 0.5*(U+V) , 0.5*(U-V)

    F21 = U'*V

    if BCs[1]%3 == 0 && BCs[2]%3 == 0 
        if Int(round(det(F21))) == 1
            U0[:,end] = -U0[:,end]
            U = Uv'*U0
            F21 = U'*V
        end 
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        r1 = sqrt(2)*real.(Q[:,2])
        ι1 = sqrt(2)*imag.(Q[:,2])
        fig , axes =  subplots(1,3)
        plot_Majorana_distribution(BCs,U0*z1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[2],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[3],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[3],true)
        axes[1].set_title("ζ1")
        axes[2].set_title("ζR2")
        axes[3].set_title("ζI2")
        for i = 1:3    
            axes[i].set_xlim(-10,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    else
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        r1 = sqrt(2)*real.(Q[:,1])
        ι1 = sqrt(2)*imag.(Q[:,1])
        fig , axes =  subplots(1,2)
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[2],true)
        axes[1].set_title("ζR1")
        axes[2].set_title("ζI1")
        for i = 1:2    
            axes[i].set_xlim(-10,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    end 
end

function plot_lowest_energy_states_two_bond_flips(BCs)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Mv = flip_bond_variable(M0,BCs,[8,8],"z")
    M2v = flip_bond_variable(Mv,BCs,[9,7],"z")
    U2v ,V2v = get_U_and_V(M2v)

    U = U2v'*U0
    V = V2v'*V0

    F21 = U'*V

    if BCs[1]%3 == 0 && BCs[2]%3 == 0 
        if Int(round(det(F21))) == -1
            U0[:,end] = -U0[:,end]
            U = U2v'*U0
            F21 = U'*V
        end 
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        r1 = sqrt(2)*real.(Q[:,1])
        ι1 = sqrt(2)*imag.(Q[:,1])
        fig , axes =  subplots(1,2)
        fig.suptitle("2 flipped link variables")
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[2],true)
        axes[1].set_title("ζR1")
        axes[2].set_title("ζI1")
        for i = 1:2    
            axes[i].set_xlim(-10,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    else
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        r1 = sqrt(2)*real.(Q[:,2])
        ι1 = sqrt(2)*imag.(Q[:,2])
        fig , axes =  subplots(1,1)
        plot_Majorana_distribution(BCs,U0*z1,"A",axes,true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes,true)
        axes.set_title("ζ1")
        axes.set_xlim(-10,10)
        axes.set_ylim(2.5,25)
        axes.set_xticklabels([])
        axes.set_yticklabels([])
        fig.suptitle("2 flipped link variables")
        figure()
        plot((U*z1).^2)
    end 
end

function plot_lowest_energy_states_three_bond_flips(BCs)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Mv = flip_bond_variable(M0,BCs,[8,8],"z")
    M2v = flip_bond_variable(Mv,BCs,[9,7],"z")
    M3v = flip_bond_variable(M2v,BCs,[10,6],"z")
    U3v ,V3v = get_U_and_V(M3v)

    U = U3v'*U0
    V = V3v'*V0

    F21 = U'*V

    if BCs[1]%3 == 0 && BCs[2]%3 == 0 
        if Int(round(det(F21))) == -1
            U0[:,end] = -U0[:,end]
            U = U3v'*U0
            F21 = U'*V
        end 
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        r1 = sqrt(2)*real.(Q[:,1])
        ι1 = sqrt(2)*imag.(Q[:,1])
        fig , axes =  subplots(1,2)
        fig.suptitle("3 flipped link variables")
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[2],true)
        axes[1].set_title("ζR1")
        axes[2].set_title("ζI1")
        for i = 1:2    
            axes[i].set_xlim(-10,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    else
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        r1 = sqrt(2)*real.(Q[:,2])
        ι1 = sqrt(2)*imag.(Q[:,2])
        fig , axes =  subplots(1,1)
        plot_Majorana_distribution(BCs,U0*z1,"A",axes,true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes,true)
        axes.set_title("ζ1")
        axes.set_xlim(-10,10)
        axes.set_ylim(2.5,25)
        axes.set_xticklabels([])
        axes.set_yticklabels([])
        fig.suptitle("3 flipped link variables")
    end 
end

function plot_lowest_energy_states_four_bond_flips(BCs)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Mv = flip_bond_variable(M0,BCs,[8,8],"z")
    M2v = flip_bond_variable(Mv,BCs,[9,7],"z")
    M3v = flip_bond_variable(M2v,BCs,[10,6],"z")
    M4v = flip_bond_variable(M3v,BCs,[11,5],"z")
    U4v ,V4v = get_U_and_V(M4v)

    #plot_bond_energies_2D_repeated_cell(M4v,BCs)

    U = U4v'*U0
    V = V4v'*V0

    F21 = U'*V

    if BCs[1]%3 == 0 && BCs[2]%3 == 0 
        if Int(round(det(F21))) == -1
            U0[:,end] = -U0[:,end]
            U = U4v'*U0
            F21 = U'*V
        end 
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        r1 = sqrt(2)*real.(Q[:,1])
        ι1 = sqrt(2)*imag.(Q[:,1])
        fig , axes =  subplots(1,2)
        fig.suptitle("4 flipped link variables")
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[2],true)
        axes[1].set_title("ζR1")
        axes[2].set_title("ζI1")
        for i = 1:2    
            axes[i].set_xlim(-10,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    else
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        r1 = sqrt(2)*real.(Q[:,2])
        ι1 = sqrt(2)*imag.(Q[:,2])
        fig , axes =  subplots(1,1)
        plot_Majorana_distribution(BCs,U0*z1,"A",axes,true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes,true)
        axes.set_title("ζ1")
        axes.set_xlim(-10,10)
        axes.set_ylim(2.5,25)
        axes.set_xticklabels([])
        axes.set_yticklabels([])
        fig.suptitle("4 flipped link variables")
    end 
end

function plot_lowest_energy_states_five_bond_flips(BCs)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Mv = flip_bond_variable(M0,BCs,[8,8],"z")
    M2v = flip_bond_variable(Mv,BCs,[9,7],"z")
    M3v = flip_bond_variable(M2v,BCs,[10,6],"z")
    M4v = flip_bond_variable(M3v,BCs,[11,5],"z")
    M5v = flip_bond_variable(M4v,BCs,[12,4],"z")
    U5v ,V5v = get_U_and_V(M5v)

    #plot_bond_energies_2D_repeated_cell(M5v,BCs)

    U = U5v'*U0
    V = V5v'*V0

    F21 = U'*V

    if BCs[1]%3 == 0 && BCs[2]%3 == 0 
        if Int(round(det(F21))) == 1
            U0[:,end] = -U0[:,end]
            U = U5v'*U0
            F21 = U'*V
        end 
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        r1 = sqrt(2)*real.(Q[:,2])
        ι1 = sqrt(2)*imag.(Q[:,2])
        fig , axes =  subplots(1,3)
        plot_Majorana_distribution(BCs,U0*z1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[2],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[3],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[3],true)
        axes[1].set_title("ζ1")
        axes[2].set_title("ζR2")
        axes[3].set_title("ζI2")
        for i = 1:3    
            axes[i].set_xlim(-7,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
        fig.suptitle("5 flipped link variables")
    else
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        z2 = real.(Q[:,2])
        r3 = sqrt(2)*real.(Q[:,3])
        ι3 = sqrt(2)*imag.(Q[:,3])
        fig , axes =  subplots(1,2)
        fig.suptitle("5 flipped link variables")
        plot_Majorana_distribution(BCs,U0*z1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*z2,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*z2,"B",axes[2],true)
        axes[1].set_title("ζ1")
        axes[2].set_title("ζ2")
        for i = 1:2    
            axes[i].set_xlim(-7,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    end 
end

function plot_lowest_energy_states_six_bond_flips(BCs)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Mv = flip_bond_variable(M0,BCs,[8,8],"z")
    M2v = flip_bond_variable(Mv,BCs,[9,7],"z")
    M3v = flip_bond_variable(M2v,BCs,[10,6],"z")
    M4v = flip_bond_variable(M3v,BCs,[11,5],"z")
    M5v = flip_bond_variable(M4v,BCs,[12,4],"z")
    M6v = flip_bond_variable(M5v,BCs,[13,3],"z")
    U6v ,V6v = get_U_and_V(M6v)

    #plot_bond_energies_2D_repeated_cell(M6v,BCs)

    U = U6v'*U0
    V = V6v'*V0

    F21 = U'*V

    if BCs[1]%3 == 0 && BCs[2]%3 == 0 
        if Int(round(det(F21))) == 1
            U0[:,end] = -U0[:,end]
            U = U6v'*U0
            F21 = U'*V
        end 
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        r1 = sqrt(2)*real.(Q[:,2])
        ι1 = sqrt(2)*imag.(Q[:,2])
        fig , axes =  subplots(1,3)
        plot_Majorana_distribution(BCs,U0*z1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[2],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[3],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[3],true)
        axes[1].set_title("ζ1")
        axes[2].set_title("ζR2")
        axes[3].set_title("ζI2")
        for i = 1:3    
            axes[i].set_xlim(-7,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
        fig.suptitle("6 flipped link variables")
    else
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        z2 = real.(Q[:,2])
        r3 = sqrt(2)*real.(Q[:,3])
        ι3 = sqrt(2)*imag.(Q[:,3])
        fig , axes =  subplots(1,2)
        fig.suptitle("5 flipped link variables")
        plot_Majorana_distribution(BCs,U0*z1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*z1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*z2,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*z2,"B",axes[2],true)
        axes[1].set_title("ζ1")
        axes[2].set_title("ζ2")
        for i = 1:2    
            axes[i].set_xlim(-7,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    end 
end

# This function is used to plot Majorana wavefunctions
function plot_Majorana_distribution(BCs,c_distribution,sublattice="A", ax2 = gca(),repeat=false)
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
        ax2.scatter(A_site_x_coords.+shift_vec[1],A_site_y_coords.+shift_vec[2].+site_shift,color = cmap.((c_distribution./maximum(c_distribution)).+0.5),s=5,zorder=2,linewidths=0)
    end 
end

#This section includes functions to work out how the ground state of a sector with visons compares to the vison free ground state

function plot_phases_of_F21_vison_pair(BCs)
    M0 = get_M0(BCs)
    U0,V0 = get_U_and_V(M0)
    Mv = flip_bond_variable(M0,BCs,[5,5],"z")
    Mv = flip_bond_variable(Mv,BCs,[6,4],"z")
    Uv , Vv = get_U_and_V(Mv)

    U = Uv'*U0
    V = Vv'*V0

    F = U'*V

    phases = eigvals(F)
    display(angle.(phases))
    

    fig = gcf()
    fig.set_size_inches(8,8)
    plot_circle(1)
    xlim(-1.1,1.1)
    ylim(-1.1,1.1)
    scatter(real.(phases),imag.(phases))

    return F , eigvecs(F)[:,1]
end 

function plot_circle(radius)
    N=100
    angles = [2*pi*θ/N for θ =0:(N)]
    x_points = radius*[cos(θ) for θ in angles]
    y_points = radius*[sin(θ) for θ in angles]

    plot(x_points,y_points)
end 

function plot_phases_of_F21_max_seperated(BCs)
    M0 = get_M0(BCs)
    U0,V0 = get_U_and_V(M0)
    Mv = get_M_for_max_seperated_visons_along_a2_boundary(BCs)
    Uv , Vv = get_U_and_V(Mv)

    U = Uv'*U0
    V = Vv'*V0

    F = U'*V

    phases = eigvals(F)
    display(angle.(phases))
    

    fig = gcf()
    fig.set_size_inches(8,8)
    plot_circle(1)
    xlim(-1.1,1.1)
    ylim(-1.1,1.1)
    scatter(real.(phases),imag.(phases))


    return F , eigvecs(F)[:,1]
end 

function plot_phases_of_F21_for_L_seperated_visons(BCs,L)
    M0 = get_M0(BCs)
    U0,V0 = get_U_and_V(M0)
    Mv = get_M0(BCs)
    for j = 0:(L-1)
        Mv = flip_bond_variable(Mv,BCs,[L+j,L-j],"z")
    end 

    Uv , Vv = get_U_and_V(Mv)

    U = Uv'*U0
    V = Vv'*V0

    F = U'*V

    phases = eigvals(F)
    display(angle.(phases))
    
    fig = gcf()
    fig.set_size_inches(8,8)
    plot_circle(1)
    xlim(-1.1,1.1)
    ylim(-1.1,1.1)
    scatter(real.(phases),imag.(phases))

    return F , eigvecs(F)[:,1]
end 

function plot_real_part_of_F21_for_L_seperated_visons_Vs_BCs(L_max,L_sep)
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
    U0,V0 = get_U_and_V(M0)
    Mv = get_M0(BCs)
    for j = 0:(L_sep-1)
            Mv = flip_bond_variable(Mv,BCs,[L_sep+j,L_sep-j],"z")
    end 

    Uv , Vv = get_U_and_V(Mv)

    U = Uv'*U0
    V = Vv'*V0

    F = U'*V
    det_F = Int(round(det(F)))

    for L = L_min:L_max
        BCs = [L,L,0]
        M0 = get_M0(BCs)
        U0,V0 = get_U_and_V(M0)
        Mv = get_M0(BCs)
        for j = 0:(L_sep-1)
            Mv = flip_bond_variable(Mv,BCs,[L_sep+j,L_sep-j],"z")
        end 

        Uv , Vv = get_U_and_V(Mv)

        U = Uv'*U0
        V = Vv'*V0

        F = U'*V

        if L%3 == 0 && Int(round(det(F))) != -(det_F)
            U0[:,end] = -U0[:,end]
            U = Uv'*U0
            F = U'*V
        end 

        cos_θ = real.(eigvals(F))[1:10]
        display(cos_θ)
        scatter(L*ones(10),cos_θ,color="b")
        sleep(0.01)
    end 
end 

function plot_states_in_GS_for_L_seperated_visons(BCs,L_sep)
   
    M0 = get_M0(BCs)
    U0,V0 = get_U_and_V(M0)
    Mv = get_M0(BCs)
    for j = 0:(L_sep-1)
        Mv = flip_bond_variable(Mv,BCs,[L_sep+j,L_sep-j],"z")
    end 

    Uv , Vv = get_U_and_V(Mv)

    U = Uv'*U0
    V = Vv'*V0

    F21 = U'*V
    nu = Int(floor((L_sep+1)/3))
    det_F = nu%2

 
    if BCs[1]%3 == 0 && Int(round(det(F21))) != -(det_F)
        U0[:,end] = -U0[:,end]
        U = Uv'*U0
        F21 = U'*V
    end 

    Q = eigvecs(F21)
    fig , axes =  subplots(1,nu)
    
    for i = 1:nu
        z = real.(Q[:,i])
        cA_distribution = U0*z
        cB_distribution = V0*z
        if nu > 1
            axes[i].set_xlim(-10,10+L_sep)
            axes[i].set_ylim(L_sep,BCs[1]+L_sep)
            plot_Majorana_distribution(BCs,cA_distribution,"A",axes[i],true)
            plot_Majorana_distribution(BCs,cB_distribution,"B",axes[i],true)
            axes[i].set_title("ζ_$i")
        else
            plot_Majorana_distribution(BCs,cA_distribution,"A",axes,true)
            plot_Majorana_distribution(BCs,cB_distribution,"B",axes,true)
            title("ζ_1")
        end 
    end 
end 

# flipping bonds on the same site 

function plot_real_part_of_F21_for_2_flipped_bonds_same_site_Vs_BCs(L_max)

    for L = 5:L_max
        BCs = [L,L,0]
        M0 = get_M0(BCs)
        U0,V0 = get_U_and_V(M0)
        Mv = flip_bond_variable(M0,BCs,[2,2],"x")
        Mv = flip_bond_variable(Mv,BCs,[2,2],"y")
        Mv = flip_bond_variable(Mv,BCs,[2,2],"z")

        Uv , Vv = get_U_and_V(Mv)

        U = Uv'*U0
        V = Vv'*V0

        F = U'*V
       
        if L%3 == 0 && Int(round(det(F))) == -1
            U0[:,end] = -U0[:,end]
            U = Uv'*U0
            F = U'*V
            cos_θ = real.(eigvals(F))[1:10]
            display(cos_θ)
            scatter(L*ones(10),cos_θ,color="r")
            sleep(0.01)
        else
            cos_θ = real.(eigvals(F))[1:10]
            display(cos_θ)
            scatter(L*ones(10),cos_θ,color="b")
            sleep(0.01)
        end 
    end 
    title("2 flipped bonds on same site")
    xlabel("system size L")
    ylabel("cos(θ)")
end 

function plot_S_same_site_bond_flips(BCs)
    """
    Plots the changes to the ground state due to flipping 3 bonds at the same site (enacting a gauge transformation)
    """
    N = BCs[1]*BCs[2]
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Mv = flip_bond_variable(M0,BCs,[0,0],"x")
    Mv = flip_bond_variable(Mv,BCs,[0,0],"y")
    Mv = flip_bond_variable(Mv,BCs,[0,0],"z")
    #plot_bond_energies_2D_repeated_cell(Mv,BCs)

    G = Int.(Matrix(I,N,N))
    G[1,1] = -1 
    U3 = G*U0
    V3 = V0

    #Uv ,Vv = get_U_and_V(Mv)

    U = U3'*U0
    V = V3'*V0

    F21 = U'*V

    if BCs[1]%3 == 0 && BCs[2]%3 == 0 
        if Int(round(det(F21))) == -1
            U0[:,end] = -U0[:,end]
            U = Uv'*U0
            F21 = U'*V
        end 
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        r1 = sqrt(2)*real.(Q[:,1])
        ι1 = sqrt(2)*imag.(Q[:,1])
        fig , axes =  subplots(1,2)
        fig.suptitle("3 flipped links on the same site")
        plot_Majorana_distribution(BCs,U0*r1,"A",axes[1],true)
        plot_Majorana_distribution(BCs,V0*r1,"B",axes[1],true)
        plot_Majorana_distribution(BCs,U0*ι1,"A",axes[2],true)
        plot_Majorana_distribution(BCs,V0*ι1,"B",axes[2],true)
        axes[1].set_title("ζR1")
        axes[2].set_title("ζI1")
        for i = 1:2    
            axes[i].set_xlim(-10,10)
            axes[i].set_ylim(2.5,25)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        end 
    else
        display(eigvals(F21)[1:5])
        Q = eigvecs(F21)
        z1 = real.(Q[:,1])
        r1 = sqrt(2)*real.(Q[:,2])
        ι1 = sqrt(2)*imag.(Q[:,2])
        fig , axes =  subplots(1,1)
        plot_Majorana_distribution(BCs,U0*z1,"A",axes,true)
        #plot_Majorana_distribution(BCs,V0*z1,"B",axes,true)
        axes.set_title("ζ1")
        axes.set_xlim(-10,10)
        axes.set_ylim(2.5,25)
        axes.set_xticklabels([])
        axes.set_yticklabels([])
        fig.suptitle("3 flipped link variables on the same site")
        
    end 
end

# plotting energy density 

function plot_energy_density_from_M_bar(M,BCs,repeat=true)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]
    N = L1*L2

    U ,V = get_U_and_V(M)
    F=U*V'
    M_bar = M*F'

    energy_density = [(1-M_bar[i,i]/1.5747386) for i=1:N]

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    #cmap = get_cmap("PuOr")
    cmap = get_cmap("seismic")

    shift_vectors = [-L2*a2-m*a1-L1*a1,-L2*a2-m*a1,-L2*a2-m*a1+L1*a1,-L1*a1,[0,0],L1*a1,L2*a2+m*a1-L1*a1,L2*a2+m*a1,L2*a2+m*a1+L1*a1]
    if repeat == false
        shift_vectors = [[0,0]]
    end 
    display(((energy_density)).+0.5)

    for shift_vec in shift_vectors
        scatter(A_site_x_coords.+shift_vec[1],A_site_y_coords.+shift_vec[2],color = cmap.((((energy_density)).*5).+0.5),zorder=2,linewidths=0)
    end 
end 

function plot_f_hopping_from_M_bar(M,BCs,site,repeat=true)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]
    N = L1*L2

    U ,V = get_U_and_V(M)
    F=U*V'
    M_bar = M*F'

    site_index , _ = convert_bond_to_site_indicies(BCs,site,"z")

    f_hopping = [abs.(M_bar[site_index,i]) for i=1:N]

    real_lattice = [n1*a1+n2*a2 for n2=0:(L2-1) for n1=0:(L1-1)] # Note that the order of the loops matters, looping over n1 happens inside loop over n2
    A_site_x_coords = [r[1] for r in real_lattice]
    A_site_y_coords = [r[2] for r in real_lattice]

    #cmap = get_cmap("PuOr")
    cmap = get_cmap("Greens")

    shift_vectors = [-L2*a2-m*a1-L1*a1,-L2*a2-m*a1,-L2*a2-m*a1+L1*a1,-L1*a1,[0,0],L1*a1,L2*a2+m*a1-L1*a1,L2*a2+m*a1,L2*a2+m*a1+L1*a1]
    if repeat == false
        shift_vectors = [[0,0]]
    end 

    for shift_vec in shift_vectors
        scatter(A_site_x_coords.+shift_vec[1],A_site_y_coords.+shift_vec[2],color = cmap.(1*f_hopping),zorder=2,linewidths=0)
    end 
end 

function plot_f_nn_hopping_2D(M,BCs)

    U,V = get_U_and_V(M)
    F = U*V'
    M_bar = M*F'

    A_sites = [n1*a1+n2*a2 for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    a1_links = [(r,r+a1) for r in A_sites]
    a2_links = [(r,r+a2) for r in A_sites]
    a1_a2_links = [(r,r+a1-a2) for r in A_sites]

    #x_link_indices, y_link_indices, z_link_indices = get_link_indices(BCs)

    link_energy_of_fluxless_system = 0.5250914141631111

    a1_link_hopping = [M_bar[convert_n1_n2_to_site_index([n1,n2],BCs),convert_n1_n2_to_site_index([n1+1,n2],BCs)] for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]
    a2_link_hopping = [M_bar[convert_n1_n2_to_site_index([n1,n2],BCs),convert_n1_n2_to_site_index([n1,n2+1],BCs)] for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]
    a1_a2_link_hopping = [M_bar[convert_n1_n2_to_site_index([n1,n2],BCs),convert_n1_n2_to_site_index([n1+1,n2-1],BCs)] for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    cmap = get_cmap("seismic") # Should choose a diverging colour map so that 0.5 maps to white 

    display(a1_link_hopping)
    a1colors = cmap.((a1_link_hopping).+0.5)
    a2colors = cmap.(a2_link_hopping.+0.5)
    a1_a2colors = cmap.(a1_a2_link_hopping.+0.5)

    a1linecollection = matplotlib.collections.LineCollection(a1_links,colors = a1colors,linewidths=2.5)
    a2linecollection = matplotlib.collections.LineCollection(a2_links,colors = a2colors,linewidths=2.5)
    a1_a2linecollection = matplotlib.collections.LineCollection(a1_a2_links,colors = a1_a2colors,linewidths=2.5)

    fig ,ax = subplots()
    ax.add_collection(a1linecollection)
    ax.add_collection(a2linecollection)
    ax.add_collection(a1_a2linecollection)
    ax.autoscale()

end 

# Adds functions for the extended Kitaev model


# sets default boundary conditions
L1 = 28
L2 = 28
m = 0
BCs = [L1,L2,m]
N=L1*L2

# This section creates two commonly used matrices M0 and Mv
M0 = get_M0(BCs)
Mv = flip_bond_variable(M0,BCs,[0,0],"z")
#M2v = flip_bond_variable(Mv, BCs,[8,8],"y")
#M3v = flip_bond_variable(M2v, BCs,[6,2],"z")
#M4v = flip_bond_variable(M3v, BCs,[7,1],"z")
#M3 = flip_bond_variable(Mv,BCs,[0,0],"y")
#M3 = flip_bond_variable(M3,BCs,[0,0],"z")

#Mv = flip_bond_variable(M0,BCs,[0,0],"z") # vison pair at [0,0] with z orientation
M_max = get_M_for_max_seperated_visons_along_a2_boundary(BCs)
display("")

U0, V0 = get_U_and_V(M0)
F0 = U0*V0'

M0_bar = M0*F0'

Uv ,Vv = get_U_and_V(Mv)
Fv = Uv*Vv'

F = (U0'*Uv)*(V0'*Vv)'

Q = eigvecs(F)
z = V0*Q[:,1]
z = z*exp(-im*angle(z[1]))
a = sqrt(2)real.(z)
b = sqrt(2)imag.(z)

O = -0.969
phi = -angle((F0*z)[1])


q = V0'*z



Λ1 = -2*(1-cos(O))*(z'*M0'*F0*z-2*real(z[1]*(F0*z)[1])) +4*sin(O)*imag(z[1]*(F0*z)[1])
display(Λ1)
Λ2 = (1-cos(O))*(transpose(z)*M0'*F0*z-2*z[1]*(F0*z)[1]) 
display(Λ2)
Eq = -2*im*sin(O)*(-z[1]*(F0)[1,:])+2*(1-cos(O))*((M0'*F0)*z-z[1]*(F0)[1,:])+Λ1*z-2*Λ2*conj.(z)


Λ = (cos(O)-1)*(2*b'*M0'*F0*a-2(b[1]*(F0*a)[1]+a[1]*(F0*b)[1]))
Λa = 0.5*(cos(O)-1)*(2*a'*M0'*F0*a-4*a[1]*(F0*a)[1])-sin(O)*(b[1]*(F0*a)[1]-a[1]*(F0*b)[1])
Λb = 0.5*(cos(O)-1)*(2*b'*M0'*F0*b-4*b[1]*(F0*b)[1])-sin(O)*(b[1]*(F0*a)[1]-a[1]*(F0*b)[1])
display(Λa)
display(Λb)
display(Λa+Λb)
display(Λ)

Eqa = (1-cos(O))*(2*M0'*F0*a-2*a[1]*F0[1,:]) + 2*sin(O)*(b[1]*F0[1,:]) + Λ*b + 2*Λa*a
Eqb = (1-cos(O))*(2*M0'*F0*b-2*b[1]*F0[1,:]) - 2*sin(O)*(a[1]*F0[1,:]) + Λ*a + 2*Λb*b 

function vary_z_phase(z0)
    for θ in LinRange(0,pi,100)
        z = z0*exp(im*θ)
        a = sqrt(2)real.(z)
        b = sqrt(2)imag.(z)
        Λ = (cos(O)-1)*(2*b'*M0'*F0*a-2(b[1]*(F0*a)[1]+a[1]*(F0*b)[1]))
        Λa = 0.5*(cos(O)-1)*(2*a'*M0'*F0*a-4*a[1]*(F0*a)[1])-sin(O)*(b[1]*(F0*a)[1]-a[1]*(F0*b)[1])
        Λb = 0.5*(cos(O)-1)*(2*b'*M0'*F0*b-4*b[1]*(F0*b)[1])-sin(O)*(b[1]*(F0*a)[1]-a[1]*(F0*b)[1])

        scatter(θ,Λ,color="b")
        scatter(θ,Λa,color="r")
        scatter(θ,Λb,color="g")
        scatter(θ,Λa+Λb,color="black")
        scatter(θ,(F0*a)[1],color="purple")
    end 
end 


#=
display(2*Fv[1,1]-2*F0[1,1]) # exact <2|V|2> - <1|V|1>
display(-2sin(O)*(a[1]*(F0*b)[1]-b[1]*(F0*a)[1])-4*sin(O/2)^2*(a[1]*(F0*a)[1]+b[1]*(F0*b)[1])) # estimate <2|V|2> - <1|V|1>
display(-2*im*sin(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])-2*(1-cos(O))*(z[1]*(F0*conj(z))[1]+conj(z)[1]*(F0*z)[1]))

display(tr(M0*(F0'-Fv'))) #exact {0^2}H^1{0^2}
display(2*sin(O/2)^2*(b'*(M0'*F0)*b+a'*(M0'*F0)*a)) #estimate in terms of a and b 
display(4*sin(O/2)^2*z'*(M0'*F0)*z) # variational estimate in terms of complex z
=#


E2 = -2*im*sin(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])+2*(1-cos(O))*(z'*(M0'*F0)*z-z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])

dE2_dO = -2*im*cos(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])+2*(sin(O))*(z'*(M0'*F0)*z-z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])

#Eq = -2*im*sin(O)*(-conj(z)[1]*(F0)[1,:])+2*(1-cos(O))*((M0'*F0)*conj(z)-conj(z)[1]*(F0)[1,:])


#display(-atan(2*z[1]^2*sin(phi)/(z'*(M0'*F0)*z-2*z[1]^2*cos(phi))))
#=
for θ in LinRange(0,2*pi,15)
    z_prime = z*exp(-im*θ)
    b_prime = sqrt(2)imag.(z_prime)
    plot_Majorana_distribution(BCs,b_prime,"B",gca(),true)
    sleep(0.01)
end 
=#

function Fix_Lambda(Λ,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 
    cos_term = 0
    for k in k_lattice
        D_k ,cos_phi_k = Delta_k(k)
        cos_term += (cos_phi_k)/(Λ + 4*sin(O/2)^2*D_k)
        Sum_term += 1/(Λ + 4*sin(O/2)^2*2*D_k)
    end 
    return (1/N)*Sum_term
end 

#=
for Λ in LinRange(0,1000,200)
    func = Fix_Lambda(Λ,BCs)
    scatter(Λ,func)
end
=#

#(2*(exp(-im*O)-im)*z[1]*F[1,:] + 2(cos(O)-1)*(M0'*F0*z))./z
