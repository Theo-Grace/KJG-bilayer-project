# This file contains functions used to study the formation of bound states between spinons and visons due to Heisenberg and gamma interactions in the Kitaev model

# Bare flux pair tight binding models 

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

function Heisenberg_hopping_no_GS_change(Q,BCs,J,K)
    N = BCs[1]*BCs[2]

    F0 = 0.52

    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Tx = J*(I(N)*(1-sign(K)*F0) +sign(K)*((U0[2,:])*(V0[1,:])'+(V0[1,:])*(U0[2,:])'))
    Ty = J*(I(N)*(1-sign(K)*F0) +sign(K)*((U0[1+BCs[1],:])*(V0[1,:])'+(V0[(BCs[1]+1),:])*(U0[1,:])'))

    T_Kitaev = abs(J-K)*(2*F0*I(N)+diagm(diag(U0'*M0*V0)))

    H_effective = Hermitian(T_Kitaev + 2*cos(dot(Q,a1))*Tx + 2*cos(dot(Q,a2))*Ty)

    return H_effective
end 

function get_Tx_Ty_TK_no_GS_change(BCs,J,K)
    N = BCs[1]*BCs[2]

    F0 = 0.52

    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Tx = J*(I(N)*(1-sign(K)*F0) +sign(K)*((U0[2,:])*(V0[1,:])'+(V0[1,:])*(U0[2,:])'))
    Ty = J*(I(N)*(1-sign(K)*F0) +sign(K)*((U0[1+BCs[1],:])*(V0[1,:])'+(V0[(BCs[1]+1),:])*(U0[1,:])'))

    TK = abs(J-K)*(2*F0*I(N)+diagm(diag(U0'*M0*V0)))

    return Tx, Ty, TK 
end 

 

function Heisenberg_hopping_GS_change(Q,BCs,J,K)
    Tx ,Ty ,TK = get_Tx_Ty_TK_for_Heisenberg_hopping(BCs,J,K)

    H_a1 = cos(dot(Q,a1))*(Tx+Tx')+im*sin(dot(Q,a1))*(Tx'-Tx)
    H_a2 = cos(dot(Q,a2))*(Ty+Ty')+im*sin(dot(Q,a2))*(Ty'-Ty)

    H_effective = Hermitian(TK + H_a1 + H_a2)

    return H_effective
end 

function get_Tx_Ty_TK_for_Heisenberg_hopping(BCs,J,K)
    N = BCs[1]*BCs[2]

    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    M_0 = flip_bond_variable(M0,BCs,[0,0],"z") # M_0 is M^u for a z oriented vison pair at r=0 

    U_0, V_0 = get_U_and_V(M_0)

    P_a1 = translate_matrix(BCs,[1,0])
    P_a2 = translate_matrix(BCs,[0,1])

    U_a1 = P_a1*U_0
    V_a1 = P_a1*V_0
    U_a2 = P_a2*U_0
    V_a2 = P_a2*V_0

    # hopping in the a1 direction 

    U_a1_0 = U_a1'*U_0
    V_a1_0 = V_a1'*V_0

    X = 0.5*(U_a1_0+V_a1_0)
    Z = inv(U_a1_0+V_a1_0)*(U_a1_0-V_a1_0)
    Y = 0.5*(U_a1_0-V_a1_0)

    overlap = abs(det(X))^(0.5)

    # Matrix elements 
    ηa1_ca1 = (U_a1 + (U_0*Z*U_a1_0'))[2,:]
    c0_η0 = (V_0 + (V_0*Z))[1,:]

    ηa1_c0 = -(V_0*U_a1_0'+(V_0*Z*U_a1_0'))[1,:]
    ca1_η0 = (U_0-(U_0*Z))[2,:]

    ηa1_η0 = U_a1_0 - U_a1_0*Z

    ca1_c0 = -(U_0*(I(N)-Z)*V_0')[2,1] 

    Tx = J*(overlap)*(ηa1_η0*(1+sign(K)*ca1_c0) + sign(K)*(ηa1_ca1*c0_η0' - ηa1_c0*ca1_η0'))

    # Hopping in the a2 direction 

    U_a2_0 = U_a2'*U_0
    V_a2_0 = V_a2'*V_0

    Z_a2 = inv(U_a2_0+V_a2_0)*(U_a2_0-V_a2_0)

    ME_a2_1 = (U_a2 + (U_0*Z_a2*U_a2_0'))[1+BCs[1],:]
    ME_a2_2 = (V_0 + (V_0*Z_a2))[1,:]

    ME_a2_3 = -(V_0*U_a2_0'+(V_0*Z_a2*U_a2_0'))[1,:]
    ME_a2_4 = (U_0-(U_0*Z_a2))[1+BCs[1],:]

    ME_a2_5 = U_a2_0 - U_a2_0*Z_a2

    ME_a2_6 = -(U_0*(I(N)-Z_a2)*V_0')[BCs[1]+1,1]

    Ty = J*(overlap)*(ME_a2_5+sign(K)*(ME_a2_1*ME_a2_2' - ME_a2_3*ME_a2_4' + ME_a2_5*ME_a2_6))
    

    #Kitaev term 
    T_Kitaev = abs(K-J)*(0.26*I(N) + 2*diagm(diag(U_0'*M_0*V_0)))

    return Tx, Ty ,T_Kitaev
end 

function get_Tx_Ty_TK_for_Heisenberg_hopping_z_hop(BCs,J,K)
    N = BCs[1]*BCs[2]

    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    M_0 = flip_bond_variable(M0,BCs,[0,0],"z") # M_0 is M^u for a z oriented vison pair at r=0 
    M_a1 = flip_bond_variable(M0,BCs,[1,0],"z")
    M_a2 = flip_bond_variable(M0,BCs,[0,1],"z")

    U_0, V_0 = get_U_and_V(M_0)

    nearest_neighbours = [[1,0],[0,1]]
    T_list = [zeros(N,N),zeros(N,N)]

    for (id,a) in enumerate(nearest_neighbours)
        P_a = translate_matrix(BCs,a)

        U_a = P_a*U_0
        V_a = P_a*V_0

        U_a_0 = U_a'*U_0
        V_a_0 = V_a'*V_0

        X = 0.5*(U_a_0+V_a_0)

        Z = inv(U_a_0+V_a_0)*(U_a_0-V_a_0)
        overlap = det(X)^(0.5)

        A_site_id = convert_n1_n2_to_site_index(a,BCs)

        ηa_ca = (U_a + (U_0*Z*U_a_0'))[A_site_id,:]
        c0_η0 = (V_0 + (V_0*Z))[1,:]

        ηa_c0 = -(V_0*U_a_0'+(V_0*Z*U_a_0'))[1,:]
        ca_η0 = (U_0-(U_0*Z))[A_site_id,:]

        ηa_η0 = U_a_0 - U_a_0*Z

        ca_c0 = -(U_0*(I(N)-Z)*V_0')[A_site_id,1] 

        display((overlap)*(1+sign(K)*ca_c0))
        display(J*(overlap)*(ηa_η0*(1+sign(K)*ca_c0) + sign(K)*(ηa_ca*c0_η0' - ηa_c0*ca_η0')))

        T_list[id]= J*(overlap)*(ηa_η0*(1+sign(K)*ca_c0) + sign(K)*(ηa_ca*c0_η0' - ηa_c0*ca_η0'))
    end 
    

    #Kitaev term 
    T_Kitaev = abs(K-J)*(0.26*I(N) + 2*diagm(diag(U_0'*M_0*V_0)))

    return T_list[1], T_list[2] ,T_Kitaev
end 

function plot_bare_flux_pair_band_structure_3D(J,K=1)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]
    
    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]
    
    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]
    
    Ek= zeros(N)
    kx_points = zeros(N)
    ky_points = zeros(N)
    cmap = get_cmap("Greens")
    
    for (id,k) in enumerate(k_lattice)
        Ek[id] = Heisenberg_hopping_no_GS_change(k,J,K)
        kx_points[id] = k[1]
        ky_points[id] = k[2]
    end 
    
    shift_k_vecs = [[0,0]] #[[0,0],g1,g2,g1+g2]
    for shift_g in shift_k_vecs
      scatter3D(kx_points.+shift_g[1],ky_points.+shift_g[2],Ek,color=cmap(Ek./maximum(Ek)))
    end
end

function plot_bare_flux_pair_band_structure(J,K=1)
    Num_k_points = 30 

    k_points = get_k_GtoKtoMtoG(Num_k_points)
    E_Q_z = zeros(Num_k_points)
    E_Q_x = zeros(Num_k_points)

    T_J = J*(0.7978 -0.6983*sign(K))

    for id = 1:Num_k_points
        Q = k_points[:,id]
        E_Q_z[id] = abs(J-K)*0.26 + 2*T_J*(cos(dot(Q,a1))+cos(dot(Q,a2)))
        E_Q_x[id] = abs(J-K)*0.26 + 2*T_J*(cos(dot(Q,a1))+cos(dot(Q,a1-a2)))
    end 

    K_index = round(2*Num_k_points/(3+sqrt(3)))
    M_index = round(3*Num_k_points/(3+sqrt(3)))
    
    axes = gca()

    axes.plot(1:Num_k_points,E_Q_z,color="red")
    axes.plot(1:Num_k_points,E_Q_x,color="blue")
    

    axes.set_xticks([1,K_index,M_index,Num_k_points])
    axes.set_xticklabels(["\$\\Gamma\$","K","M","\$\\Gamma\$"])
    axes.set_ylabel("Energy")
    axes.vlines([1,K_index,M_index,Num_k_points],-(0.1),1.5,linestyle="dashed")
    
end

function plot_bare_flux_pair_band_structure_via_M_tilde(BCs,J,K=1)
    Num_k_points = 30 
    Num_ex = 50

    k_points = get_k_GtoKtoMtoG(Num_k_points)
    E_Q_z = zeros(Num_k_points)
    E_Q_excited = zeros(Num_k_points,Num_ex)

    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    E0 = abs(K-J)*sum(svd(M0).S)

    Mv = flip_bond_variable(M0,BCs,[0,0],"z")
    Uv ,Vv = get_U_and_V(Mv)

    F = (U0'*Uv)*(V0'*Vv)'

    for id = 1:Num_k_points
        Q = k_points[:,id]

        M_eff = Heisenberg_hopping_M(BCs,J,K,Q,F)
        E_ex_eff = 2*reverse(svd(M_eff).S)

        E_Q_z[id] = -(tr(M_eff*F')-E0)
        E_Q_excited[id,:] = E_ex_eff[1:Num_ex] .+ E_Q_z[id]

        display(E_Q_z[id])
    end 

    K_index = round(2*Num_k_points/(3+sqrt(3)))
    M_index = round(3*Num_k_points/(3+sqrt(3)))
    
    axes = gca()

    axes.plot(1:Num_k_points,E_Q_z,color="green")
    for ex_id = 1:Num_ex
        axes.plot(1:Num_k_points,E_Q_excited[:,ex_id],color="blue")
    end
    

    axes.set_xticks([1,K_index,M_index,Num_k_points])
    axes.set_xticklabels(["\$\\Gamma\$","K","M","\$\\Gamma\$"])
    axes.set_ylabel("Energy")
    axes.vlines([1,K_index,M_index,Num_k_points],-(0.1),1.5,linestyle="dashed")
    
end

function get_k_GtoKtoMtoG(Num_points)
    """
    Used to create a list of k vectors along the path G to K to M to G in the Brillouin zone. 
    Takes:
    - Num_points which is the number of k vectors to sample along the path

    returns:
    - a 2xNum_points matrix, where each column is a k vector
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

function get_z_Heisenberg_bandstructure_GtoKtoMtoG(J,K,Num_k_points)
    BCs = [35,35,0]

    kGtoKtoMtoG = get_k_GtoKtoMtoG(Num_k_points)

    bands_GtoKtoMtoG = zeros(BCs[1]*BCs[2],Num_k_points)

    Tx ,Ty ,TK = get_Tx_Ty_TK_for_Heisenberg_hopping_z_hop(BCs,J,K)
    
    for id = 1:Num_k_points
        Q = kGtoKtoMtoG[:,id]
        H_a1 = cos(dot(Q,a1))*(Tx+Tx')+im*sin(dot(Q,a1))*(Tx'-Tx)
        H_a2 = cos(dot(Q,a2))*(Ty+Ty')+im*sin(dot(Q,a2))*(Ty'-Ty)
        H = TK + H_a1 + H_a2
        bands_GtoKtoMtoG[:,id] = eigvals(H)
    end

    return bands_GtoKtoMtoG
end

function get_z_no_GS_change_bandstructure_GtoKtoMtoG(J,K,Num_k_points)
    BCs = [35,35,0]

    kGtoKtoMtoG = get_k_GtoKtoMtoG(Num_k_points)

    bands_GtoKtoMtoG = zeros(BCs[1]*BCs[2],Num_k_points)

    Tx ,Ty ,TK = get_Tx_Ty_TK_no_GS_change(BCs,J,K)
    for id = 1:Num_k_points
        Q = kGtoKtoMtoG[:,id]
        H_a1 = cos(dot(Q,a1))*(Tx+Tx')+im*sin(dot(Q,a1))*(Tx'-Tx)
        H_a2 = cos(dot(Q,a2))*(Ty+Ty')+im*sin(dot(Q,a2))*(Ty'-Ty)
        H = Hermitian(TK + H_a1 + H_a2)
        bands_GtoKtoMtoG[:,id] = eigvals(H)
    end

    return bands_GtoKtoMtoG
end

function plot_bands_GtoKtoMtoG(kGtoKtoMtoG,bands_GtoKtoMtoG,axes)
    """
    An alternative way to plot the bandstructure (Faster than plot_bands_G_to_K_to_M_to_G) that takes:
    - kGtoKtoMtoG a prepared list of k vectors along the path G to K to M to G as a 2xN matrix where N is the number of k vectors sampled along the path
    - bands_GtoKtoMtoG a list of energies at the corresponding k vectors as a 16xN matrix
    - an axis object axes used to add features to the plot such as labels. 
    """
    Num_k_points = Int(size(kGtoKtoMtoG)[2])

    K_index = round(2*Num_k_points/(3+sqrt(3)))
    M_index = round(3*Num_k_points/(3+sqrt(3)))
    
    E_max = 2 
    for i = 1:300
        axes.plot(1:Num_k_points,bands_GtoKtoMtoG[i,:],color="black")
    end

    axes.set_xticks([1,K_index,M_index,Num_k_points])
    axes.set_xticklabels(["\$\\Gamma\$","K","M","\$\\Gamma\$"])
    axes.set_ylabel("Energy")
    axes.vlines([1,K_index,M_index,Num_k_points],-(0.1),1.1*E_max,linestyle="dashed")
    axes.set_ylim([-0.05,E_max])
end

# exact hopping calculations via effective M calculation

function Heisenberg_hopping_M(BCs,J,K,Q,F)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)

    Δ = diagm(svd(M0).S)

    P_a1 = translate_matrix(BCs,[1,0])
    T_a1 = U0'*P_a1*U0

    P_a2 = translate_matrix(BCs,[0,1])
    T_a2 = U0'*P_a2*U0

    A_a1 = 0.5*(T_a1*F'+F'*T_a1)
    A_a2 = 0.5*(T_a2*F'+F'*T_a2)

    F_tilde = abs(K-J)*F + J*cos(dot(Q,a1))*abs(det(A_a1))^(0.5)*(U0'*V0-sign(K)*inv(A_a1)) + J*cos(dot(Q,a2))*abs(det(A_a2))^(0.5)*(U0'*V0-sign(K)*inv(A_a2))

    M_tilde = abs(K-J)*Δ -2*(U0*F_tilde*F')[1,:]*V0[1,:]'

    return M_tilde
end 

function iterate_M_Heisenberg(BCs,J,K,Q,F_init)
    M_init = Heisenberg_hopping_M(BCs,J,K,Q,F_init)

    M = M_init

    for it_num = 1:12
        U ,V = get_U_and_V(M)

        F = U*V'

        M = Heisenberg_hopping_M(BCs,J,K,Q,F)
        display(svd(M).S)
    end 

    return M 
end 



# Spectrum at High symmetry points 

function plot_spectrum_at_high_symmetry_point(J_range,point,K=1,GS_change=true)
    BCs = [29,29,0]
    N = BCs[1]*BCs[2]

    if point == "Gamma"
        k_point = [0,0]
        title_str = "Γ point"
    elseif point == "M"
        k_point = 0.5*(g1+g2)
        title_str = "M point"
    elseif point == "M_prime"
        k_point = 0.5*(g1)
        title_str = "M' point"
    elseif point == "K"
        k_point = 0.5*(g1-g2)
        title_str = "K point"
    end 

    Num_J = 20

    E_matrix = zeros(Num_J,N)
    J_points = LinRange(J_range[1],J_range[2],Num_J)

    for (id,J) in enumerate(J_points)
        if GS_change == true 
            H_eff = Heisenberg_hopping_GS_change(k_point,BCs,J,K)
        else
            H_eff = Heisenberg_hopping_no_GS_change(k_point,BCs,J,K)
        end 
        E_matrix[id,:] = eigvals(H_eff)
    end 

    Num_states = 10 

    if GS_change == true
        cmap = get_cmap("Reds")
    else
        cmap = get_cmap("Greens")
    end 

    for ex_id = 1:Num_states
        plot(J_points,E_matrix[:,ex_id],color=cmap(1-0.5*(ex_id/Num_states)))
    end 
    xlabel("J/|K|")
    ylabel("ϵ/|K|")
    title(title_str)
end 

# bound state Majorana wavefunction

function calculate_bound_state_wave_function(BCs,J,K)
    """
    Calculate the real space wavefunction ψ_r at the Γ point 
                Υ = ψ_r c_r^A 
    """
    M0 = get_M0(BCs)
    Mv = flip_bond_variable(M0,BCs,[0,0],"z")
    Uv , Vv = get_U_and_V(Mv)

    H_eff = Heisenberg_hopping_GS_change([0,0],BCs,J,K)
    E = eigvals(H_eff)
    display(E[1:5])
    psi_0z = real.(eigvecs(H_eff)[:,1])

    psi = Uv*psi_0z

    return psi 
end 

function plot_real_space_bound_state_wavefunction(BCs,J,K)
    psi = calculate_bound_state_wave_function(BCs,J,K)

    M0 = get_M0(BCs)
    Mv = flip_bond_variable(M0,BCs,[0,0],"z")
    Uv , Vv = get_U_and_V(Mv)

    psi_B = Vv*Uv'*psi


    plot_Majorana_distribution(BCs,psi,"A",gca(),true)
    plot_Majorana_distribution(BCs,psi_B,"B",gca(),true)
end 

function plot_k_space_bound_state_wavefunction(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    real_lattice = [n1*a1+n2*a2 for n1 = 0:(L1-1) for n2 = 0:(L2-1)]

    ψ_r = calculate_bound_state_wave_function(BCs,-0.33,1)

    kx_points = zeros(N)
    ky_points = zeros(N)

    cmap = get_cmap("seismic")

    psi_k_plus = zeros(N)
    psi_k_minus = zeros(N)

    for (id,k) in enumerate(k_lattice)
        for (r_id,r) in enumerate(real_lattice)
            psi_k_plus[id] += cos(dot(k,r))*ψ_r[r_id]
            psi_k_minus[id] += sin(dot(k,r))*ψ_r[r_id]
        end 
        kx_points[id] = k[1]
        ky_points[id] = k[2]
    end 

    psi_plus_max = maximum(psi_k_plus)
    psi_minus_max = maximum(psi_k_minus)

    norm_k_plus = matplotlib.colors.Normalize(-psi_plus_max,psi_plus_max,true)
    norm_k_minus = matplotlib.colors.Normalize(-psi_minus_max,psi_minus_max,true)

    shift_k_vecs = [[0,0],g1,g2,g1+g2]
    for shift_g in shift_k_vecs
        #scatter3D(kx_points.+shift_g[1],ky_points.+shift_g[2],psi_k_plus,color=cmap(norm_k_plus(psi_k_plus)))
        scatter(kx_points.+shift_g[1],ky_points.+shift_g[2],color=cmap(norm_k_plus(psi_k_plus)))
        #scatter(kx_points.+shift_g[1],ky_points.+shift_g[2],color=cmap(norm_k_minus(psi_k_minus)))
        #scatter3D(kx_points.+shift_g[1],ky_points.+shift_g[2],psi_k_minus,color=cmap(psi_k_minus./maximum(psi_k_minus)))
    end
end

# Calculation of interlayer hopping for bound states 

function calculate_bound_state_interlayer_hopping_parameter(BCs,J,K)
    N = BCs[1]*BCs[2]

    ψ_r = calculate_bound_state_wave_function(BCs,J,K)

    M0 = get_M0(BCs)
    Mv = flip_bond_variable(M0,BCs,[0,0],"z")

    U0 ,V0 = get_U_and_V(M0)
    Uv ,Vv = get_U_and_V(Mv)

    U = U0'*Uv
    V = V0'*Vv

    F = U*V'
    Z = inv(I(N)+F)*(I(N)-F)

    overlap = abs(det(0.5*(U+V)))^(0.5)

    A_site_hopping = overlap*(ψ_r[1] +((U0*Z*U0')*ψ_r)[1])
    display(A_site_hopping)
    B_site_hopping =  overlap*((V0*U0' +(V0*Z*U0'))*ψ_r)[1]
    display(B_site_hopping)

    hopping_parameter = (A_site_hopping^2 + B_site_hopping^2)

    return hopping_parameter
end 

function plot_interlayer_bound_state_hopping_vs_BCs(L_max,J,K)

    L_start = 5

    gca().set_ylim([0,1.5])
    xlabel("1/L")

    b_hop = zeros(Int(ceil((L_max-L_start)/3))+1)
    b_L = zeros(Int(ceil((L_max-L_start)/3))+1)
    b_id=1

    g_hop = zeros(Int(ceil((L_max-L_start)/3))+1)
    g_L = zeros(Int(ceil((L_max-L_start)/3))+1)
    g_id=1

    r_hop = zeros(Int(ceil((L_max-L_start)/3))+1)
    r_L = zeros(Int(ceil((L_max-L_start)/3))+1)
    r_id=1

    for L = L_start:L_max
        BCs = [L,L,0]
        hop_parameter = calculate_bound_state_interlayer_hopping_parameter(BCs,J,K)
 
        if L%3 ==0 
            scatter(1/L,hop_parameter,color="r")
            r_L[r_id] = 1/L
            r_hop[r_id] = hop_parameter
            r_id+=1
        elseif L%3 == 1
            scatter(1/L,hop_parameter,color="b")
            b_L[b_id] = 1/L
            b_hop[b_id] = hop_parameter
            b_id+=1
        else
            scatter(1/L,hop_parameter,color="g")
            g_L[g_id] = 1/L
            g_hop[g_id] = hop_parameter
            g_id+=1
        end 

        sleep(0.01)
        display(L)
    end 
    if L_max%3==0
        legend(["L%3=0","L%3=1","L%3=2"])
    elseif L_max%3==1
        legend(["L%3=1","L%3=2","L%3=0"])
    else
        legend(["L%3=2","L%3=0","L%3=1"])
    end 
    b_data = [b_L,b_hop]
    r_data = [r_L,r_hop]
    g_data = [g_L,g_hop]

    plot_finite_scaling(b_data,"b")
    plot_finite_scaling(g_data,"g")

    return b_data , g_data , r_data
end




#=
# Variational calculations 

BCs = [26,26,0]
N = BCs[1]*BCs[2]
M0 = get_M0(BCs)
Mv = flip_bond_variable(M0,BCs,[0,0],"z")

U0 ,V0 = get_U_and_V(M0)
Uv, Vv = get_U_and_V(Mv)

E0 = (svd(M0).S)

F = U0'*Uv*Vv'*V0
Z = inv(I(N)+F)*(I(N)-F)

F0 = U0*V0'

Q = -im*eigvecs(Hermitian(im*Z))
Z_eig = -im*eigvals(Hermitian(im*Z))
F_eig = -(Z_eig.-1)./(Z_eig.+1)
exp_iθ = F_eig[1]

approx_eigvals = ones(Complex{Float64},N)
approx_eigvals[1] = exp_iθ
approx_eigvals[end] = conj(exp_iθ)

Q = eigvecs(Hermitian(im*Z))
z = Q[:,1] 

O = angle.(eigvals(F))[1]

Λ2 = -2(cos(O)-1)*(transpose(z)*U0'*M0*V0*z-2*((U0*z)[1]*(V0*z)[1])) 

ϕ = angle(Λ2) 

z = -im*exp(-im*ϕ/2)*z
z = V0*z

# Lagrange multipliers
Λ1_tilde = sin(O/2)*(transpose(z)*M0'*F0*z-2*z[1]*(F0*z)[1])
angle(Λ1_tilde) 
Λ2_tilde = sin(O/2)*(z'*M0'*F0*z) - 2*imag(z'[1]*(F0*z)[1]*exp(im*O/2)) 

Λa = (Λ2_tilde + abs(Λ1_tilde))/sin(-O/2)  
Λb = (Λ2_tilde - abs(Λ1_tilde))/sin(-O/2)  

Aa = sqrt(2)*abs(z[1])*sin((phi-O)/2)/sin(-O/2) 
Ab = sqrt(2)*abs(z[1])*cos((phi-O)/2)/sin(-O/2) 

z = V0'*z
a = sqrt(2)*real.(z)   
b = sqrt(2)*imag.(z)   

#=
Λa = 1.47 #1.5396
Λb = 0.275 #0.2995
θ = -0.9861

a = (U0[1,:]+V0[1,:])./(E0.+Λa)
b = -(U0[1,:]-V0[1,:])./(E0.+Λb)

Aa = 1/(sqrt(a'*a))
Ab = 1/sqrt(b'*b)

a = Aa*a
b = Ab*b 
=#

Eq_a = Λa*(V0*a)+V0*diagm(E0)*a-Aa*F0[1,:]

F_approx = real.(Q*diagm(approx_eigvals)*Q')
F_approx_2 = I(N) + (cos(O)-1)*(a*a'+b*b') + sin(O)*(a*b'-b*a')

test_psi = zeros(N)
test_psi[(N-3):end] = [1,1,1,1]
test_psi = test_psi./sqrt(test_psi'*test_psi)

# <0^z|Υ H^{u_0} Υ|0^z>

H0_exp_exact = 2*test_psi'*F*diagm(E0)*test_psi
H0_exp_approx = 2*test_psi'*F_approx*diagm(E0)*test_psi
H0_exp_approx_2 = 2*test_psi'*diagm(E0)*test_psi +
4*Aa*(U0*test_psi)[1]*((cos(O)-1)*(a'*test_psi)-sin(O)*(b'*test_psi)) -
2*(cos(O)-1)*(Λa*(a'*test_psi)^2 + Λb*(b'*test_psi)^2) -
2*sin(O)*(a'*test_psi)*(b'*test_psi)*(Λa-Λb) 

H0_exp_approx_3 = 2*test_psi'*diagm(E0)*test_psi +
2*(a'*test_psi)*((cos(O)-1)*(a'*diagm(E0)*test_psi)+sin(O)*(b'*diagm(E0)*test_psi)) +
2*(b'*test_psi)*((cos(O)-1)*(b'*diagm(E0)*test_psi)-sin(O)*(a'*diagm(E0)*test_psi)) 

# Ignoring changes to GS 
H0_exp_0 = 2*test_psi'*diagm(E0)*test_psi


# static vison expectation value <0^z|Υ V Υ|0^z>

V_exact = -4*(U0*test_psi)[1]*(V0*F'*test_psi)[1]
V_approx = -4*(U0*test_psi)[1]*(V0*F_approx'*test_psi)[1]
# Ignoring the change to GS 
V0 = -4*(U0*test_psi)[1]*(V0*test_psi)[1]
# Including the change to GS 
V_approx_2 = -4*(U0*test_psi)[1]*(V0*test_psi)[1] -
              4*(U0*test_psi)[1]*((cos(O)-1)*(U0*a)[1]-sin(O)*(U0*b)[1])*(a'*test_psi) +
              4*(U0*test_psi)[1]*((cos(O)-1)*(U0*b)[1]+sin(O)*(U0*a)[1])*(b'*test_psi) 

=#