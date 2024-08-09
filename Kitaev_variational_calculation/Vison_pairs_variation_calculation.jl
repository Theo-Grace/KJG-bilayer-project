# These are numerical calculations used to calculate the Matter sector ground state for the case of open flux pairs 
# The functions used come from Bilayer_Variationalcalc_2.jl 

# Sets the boundary conditions 
L1 = 20
L2 = 20
m = 0
BCs = [L1,L2,m]
N=L1*L2

# gets the Matter sector Hamiltonian for fluxless and pair flux states 
M0 = get_M0(BCs)

Mv = flip_bond_variable(M0,BCs,[0,0],"z")

# Singular value decomposition and polar matrices
U0 ,V0 = get_U_and_V(M0)
Uv ,Vv = get_U_and_V(Mv)
Fv = Uv*Vv'
F0 = U0*V0'

F = (U0'*Uv)*(V0'*Vv)'
Z = inv(I(L1*L2)+F)*(I(L1*L2)-F)

function plot_2nd_order_Z_approx_vs_BCs(L_max)
    gca().set_ylim([-1.5,0.1])
    for L = 5:L_max 
        BCs = [L,L,0]
        M0 = get_M0(BCs)
        Mv = flip_bond_variable(M0,BCs,[0,0],"z")

        U0 ,V0 = get_U_and_V(M0)
        Uv , Vv = get_U_and_V(Mv)

        F = (U0'*Uv)*(V0'*Vv)'
        Z = inv(I(L^2)+F)*(I(L^2)-F)

        E0 = diagm(svd(M0).S)

        δE0 = tr(E0*F')-tr(E0)
        δE0_approx = 2*tr(E0*(Z^2))
        #δE = -2*tr(E0*Z*inv(I(L^2)+Z))

        scatter(1/L,δE0,color="r")
        scatter(1/L,δE0_approx,color="b")
        #scatter(1/L,δE,color="g")
        sleep(0.05)
    end 
end     

Q = eigvecs(Hermitian(im*Z))
z = Q[:,1] 

O = angle.(eigvals(F))[1]

Λ2 = -2(cos(O)-1)*(transpose(z)*U0'*M0*V0*z-2*((U0*z)[1]*(V0*z)[1])) 

ϕ = angle(Λ2) 

z = im*exp(-im*ϕ/2)*z
z = V0*z
a = sqrt(2)*real.(z) 
b = sqrt(2)*imag.(z)  

phi = angle(z[1]/(F0*z)[1])  

# Estimates of Potential 

V_exact = (2*Fv[1,1]-2*F0[1,1]) # exact <2|V|2> - <1|V|1>
V_est = (-2sin(O)*(a[1]*(F0*b)[1]-b[1]*(F0*a)[1])-4*sin(O/2)^2*(a[1]*(F0*a)[1]+b[1]*(F0*b)[1])) # estimate <2|V|2> - <1|V|1>
display(-2sin(O)*(a[1]*(F0*b)[1]-b[1]*(F0*a)[1]))
display(-4*sin(O/2)^2*(a[1]*(F0*a)[1]+b[1]*(F0*b)[1]))
display(-4*sin(O/2)^2*(a[1]*(F0*a)[1]))
display(-4*sin(O/2)^2*(b[1]*(F0*b)[1]))

V_complex_est = -2*im*sin(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])-2*(1-cos(O))*(z[1]*(F0*conj(z))[1]+conj(z)[1]*(F0*z)[1])

#Estimates of H0 expectation values

H0_exp_exact = tr(M0*(F0'-Fv')) #exact {0^2}H^1{0^2}
H0_exp_est = (2*sin(O/2)^2*(b'*(M0'*F0)*b+a'*(M0'*F0)*a)) #estimate in terms of a and b 
H0_comp_est = (4*sin(O/2)^2*z'*(M0'*F0)*z) # variational estimate in terms of complex z

# Gap to pairs

Δ_exact = 2*F0[1,1] + H0_exp_exact + V_exact 
Δ_est = 2*F0[1,1] + H0_exp_est + V_est 

# Lagrange multipliers
Λ1_tilde = sin(O/2)*(transpose(z)*M0'*F0*z-2*z[1]*(F0*z)[1])
angle(Λ1_tilde) 
Λ2_tilde = sin(O/2)*(z'*M0'*F0*z) - 2*imag(z'[1]*(F0*z)[1]*exp(im*O/2))

Λa = (Λ2_tilde + abs(Λ1_tilde))/sin(-O/2)
Λb = (Λ2_tilde - abs(Λ1_tilde))/sin(-O/2) 

Aa = sqrt(2)*abs(z[1])*sin((phi-O)/2)/sin(-O/2)
Ab = sqrt(2)*abs(z[1])*cos((phi-O)/2)/sin(-O/2)

# Equations from Minimisation of E2
Eq_a = Λa*a + M0'*F0*a - Aa*F0[1,:]
Eq_b = Λb*b + M0'*F0*b + Ab*F0[1,:]

ak = V0'*a
bk = V0'*b

Eq_ak = Λa*ak + V0'*M0'*U0*ak -Aa*(U0[1,:])-Aa*V0[1,:]
Eq_bk = Λb*bk + V0'*M0'*U0*bk +Ab*(U0[1,:]-V0[1,:])


display((a[1]-Aa)*tan(O/2)-b[1]) # Equation from minimising E with respect to Λa
display((b[1]+Ab)*tan(O/2)+a[1]) # Equation from minimising E with respect to Λb
display(-a[1]^2+a[1]*Aa-b[1]^2-b[1]*Ab)


# Variational calculation

function get_Aa(Λa)
    BCs = [1000,1000,0]
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 
    Inf_sum = 0 

    for k in k_lattice
        D_k ,cos_phi_k = Delta_k(k)
        Sum_term += (1+cos_phi_k)/(Λa + D_k)^2   
        Inf_sum += (1+cos_phi_k)
    end 
    #display(Inf_sum/N)
    Sum_term = (2/N)*Sum_term
    Aa = (Sum_term)^(-0.5)
    return Aa
end 

function get_Ab(Λb)
    BCs = [1000,1000,0]
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 

    for k in k_lattice
        D_k ,cos_phi_k = Delta_k(k)
        Sum_term += (1-cos_phi_k)/(Λb + D_k)^2   
    end 

    Sum_term = (2/N)*Sum_term
    Ab = (Sum_term)^(-0.5)
    return Ab
end 

function get_Ua(Λa)
    BCs= [1000,1000,0]
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 
    cos_sum = 0 

    for k in k_lattice
        D_k ,cos_phi_k = Delta_k(k)
        Sum_term += (1+cos_phi_k)/(Λa + D_k)  
        cos_sum +=cos_phi_k
    end 

    Sum_term = (1/N)*Sum_term
    Aa = get_Aa(Λa)
    Ua = Aa*Sum_term
    return Ua
end

function get_Ub(Λb)
    BCs = [1000,1000,0]
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 

    for k in k_lattice
        D_k ,cos_phi_k = Delta_k(k)
        Sum_term += (1-cos_phi_k)/(Λb + D_k)  
    end 

    Sum_term = (1/N)*Sum_term
    Ab = get_Ab(Λb)
    Ub = Ab*Sum_term
    return Ub
end

function plot_ak_space_wavefunction(Λa,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    ak = zeros(N)
    Aa = get_Aa(Λa)
    kx_points = zeros(N)
    ky_points = zeros(N)
    cmap = get_cmap("Greens")

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        ak[id] = (1+cos_phi_k)/(Λa + D_k)  
        kx_points[id] = k[1]
        ky_points[id] = k[2]
    end 

    shift_k_vecs = [[0,0],g1,g2,g1+g2]
    for shift_g in shift_k_vecs
        scatter3D(kx_points.+shift_g[1],ky_points.+shift_g[2],ak,color=cmap(ak./maximum(ak)))
    end
end

function plot_ak_space_wavefunction_xbond(Λa,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    ak = zeros(N)
    Aa = get_Aa(Λa)
    kx_points = zeros(N)
    ky_points = zeros(N)
    cmap = get_cmap("Greens")

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        ak[id] = (1+cos(phi_k-dot(k,a1)))/(Λa + D_k)  
        kx_points[id] = k[1]
        ky_points[id] = k[2]
    end 

    shift_k_vecs = [[0,0],g1,g2,g1+g2]
    for shift_g in shift_k_vecs
        scatter3D(kx_points.+shift_g[1],ky_points.+shift_g[2],ak,color=cmap(ak./maximum(ak)))
    end
end

function plot_bk_space_wavefunction(Λb,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    bk = zeros(N)
    Ab = get_Ab(Λb)
    kx_points = zeros(N)
    ky_points = zeros(N)
    cmap = get_cmap("Greens")

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        bk[id] = (1-cos_phi_k)/(Λb + D_k)  
        kx_points[id] = k[1]
        ky_points[id] = k[2]
    end 
    bk=Ab.*bk

    shift_k_vecs = [[0,0],g1,g2,g1+g2]
    for shift_g in shift_k_vecs
        scatter3D(kx_points.+shift_g[1],ky_points.+shift_g[2],bk,color=cmap(bk./maximum(bk)))
    end
end

function calculate_ar(Λa,r,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    
    Aa = get_Aa(Λa)
    Sum_term = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        Sum_term  += (cos(dot(k,r))+cos(dot(k,r)+phi_k-dot(k,a1)))/(Λa + D_k)  
    end 

    ar = (Aa/N)*Sum_term
    return ar
end 

function calculate_br(Λb,r,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    
    Ab = get_Ab(Λb)
    Sum_term = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        Sum_term  += (cos(dot(k,r))-cos(dot(k,r)+phi_k-dot(k,a1)))/(Λb + D_k)  
    end 

    br = (Ab/N)*Sum_term
    return br
end 

function plot_ar_real_space_wavefunction(Λa,BCs)
    L1 = 10
    L2 = 10
    N = (2*L1+1)*(2*L2+1)
    real_lattice = [n1*a1+n2*a2 for n1 = -L1:L1 for n2 = -L2:L2]

    rx_points = zeros(N)
    ry_points = zeros(N)
    ar_points = zeros(N)

    cmap = get_cmap("seismic")

    for (id,r) in enumerate(real_lattice)
        rx_points[id] = r[1]
        ry_points[id] = r[2]
        ar_points[id] = calculate_ar(Λa,r,BCs)
    end 

    #scatter3D(rx_points,ry_points,ar_points,color = cmap((8*ar_points./maximum(2*ar_points)).+0.5))
    scatter(rx_points,ry_points,color = cmap((ar_points./maximum(2*ar_points)).+0.5))

    return ar_points
end

function plot_br_real_space_wavefunction(Λb,BCs)
    L1 = 20
    L2 = 20
    N = (2*L1+1)*(2*L2+1)
    real_lattice = [n1*a1+n2*a2 for n1 = -L1:L1 for n2 = -L2:L2]

    rx_points = zeros(N)
    ry_points = zeros(N)
    br_points = zeros(N)

    cmap = get_cmap("seismic")

    for (id,r) in enumerate(real_lattice)
        rx_points[id] = r[1]
        ry_points[id] = r[2]
        br_points[id] = calculate_br(Λb,r,BCs)
    end 

    #scatter3D(rx_points,ry_points,br_points,color = cmap((8*br_points./maximum(2*br_points)).+0.5))

    scatter(rx_points,ry_points,color = cmap((br_points./maximum(2*br_points)).+0.5))
    return br_points
end

function find_Lambda_b_from_Lambda_a(Λa,Λb_start=0.215)
    Ua = get_Ua(Λa)
    Aa = get_Aa(Λa)

    func_a = Ua*Aa-Ua^2
    func_b = 0 
    Λb = Λb_start

    while func_a - func_b > 0 
        Ub = get_Ub(Λb)
        Ab = get_Ab(Λb)
        func_b = Ub^2+Ub*Ab
        Λb +=0.0001
        display(Λb)
    end 
    return Λb
end 

function plot_E2_vs_Lambda_a(Λa_range)
    xlabel("Λa")
    ylabel("Pairing Energy δ")
    for Λa in LinRange(Λa_range[1],Λa_range[2],30)
        #Ab_point = get_Ab(Λ)
        Aa = get_Aa(Λa)
        Ua = get_Ua(Λa)
        #Ub_point = get_Ub(Λ)
        #func_a = Ua_point*Aa_point-Ua_point^2
        #func_b = Ub_point^2+Ub_point*Ab_point
    
        Λb= find_Lambda_b_from_Lambda_a(Λa)
        #scatter(Λ,Λb_point)
        Ub = get_Ub(Λb)
        Ab = get_Ab(Λb)
    
        tan_O_2 = -Ub/(Aa-Ua)
        O = 2*atan(tan_O_2)
        scatter(Λa,O,color="r")
    
        #display(-2sin(O)*(a[1]*(F0*b)[1]-b[1]*(F0*a)[1])-4*sin(O/2)^2*(a[1]*(F0*a)[1]+b[1]*(F0*b)[1]))
        V_point = 4*sin(O)*Ub*Ua - 4*sin(O/2)^2*(Ua^2-Ub^2)
    
        #display(2*sin(O/2)^2*(b'*(M0'*F0)*b+a'*(M0'*F0)*a))
        H_point = 2*sin(O/2)^2*(2*Aa*Ua-Λa+2*Ab*Ub-Λb)
    
        #scatter(Λ,V_point,color="b")
        #scatter(Λ,H_point,color="r")
        scatter(Λa,V_point+H_point,color="g")
        sleep(0.05)
        #scatter(Λ,Ub_point,color="b")
        #scatter(Λ,tan_O_2)
        #scatter(Λ,Ua_point,color="r")
        #scatter(Λ,func_a)
        #scatter(Λ,func_b)
        #scatter(Λ,Aa_point,color="g")
        #scatter(Λ,Ab_point)
        #scatter(Λ,4*Ua_point*Ub_point^2/(Aa_point-Ua_point))
        #scatter(Λ,Ub_point/(Ab_point+Ub_point))
    end
end 

function plot_min_theta_equation(Λa_range)
    Num_points = 10

    Λa_list = LinRange(Λa_range[1],Λa_range[2],Num_points)
    Eq_1 = zeros(Num_points)
    Eq_2 = zeros(Num_points)
    Λb = 0.285

    for (id,Λa) in enumerate(Λa_list)
        display(id)

        Aa = get_Aa(Λa)
        Ua = get_Ua(Λa)
    
        Λb= find_Lambda_b_from_Lambda_a(Λa,Λb)
        Ub = -get_Ub(Λb)
        Ab = get_Ab(Λb)

    
        tan_O_2 = -Ub/(Aa-Ua)
        O = 2*atan(tan_O_2)
        display(O)

        Eq_1[id] = -sin(O)*(Λa+Λb)
        Eq_2[id] = 4*Ua*Ub

        scatter(Λa,-sin(O)*(Λa+Λb),color="g")
        scatter(Λa,4*Ua*Ub,color="r")
        sleep(0.05)
    end
    plot(Λa_list,Eq_1,color="g")
    plot(Λa_list,Eq_2,color="r")

    xlabel("Λa")
    ylabel("Pairing Energy δ")

    return Λa_list, Eq_1, Eq_2
end 

# functions to calculate overlaps once Lambda_a A_a etc are fixed by minimisation. 

function calculate_az_ax_overlap(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Aa = 1.703
    Λa = 1.54

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (1-cos(dot(k,a1)))/(Λa + D_k)^2 
    end 

    overlap = 1- (Aa^2/N)*overlap_sum

    return overlap 
end 

function calculate_bz_bx_overlap(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.199
    Λb = 0.301

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (1-cos(dot(k,a1)))/(Λb + D_k)^2 
    end 

    overlap = 1- (Ab^2/N)*overlap_sum

    return overlap 
end 

function calculate_bz_ax_overlap(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.199
    Λb = 0.301
    Aa = 1.703
    Λa = 1.54

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (1-cos(dot(k,a1)))/((Λb + D_k)*(Λa + D_k))
    end 

    overlap = 1- (Ab*Aa/N)*overlap_sum

    return overlap 
end 

function calculate_Vb_z_a2(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.199
    Λb = 0.301

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(phi_k-dot(k,a2))-cos(dot(k,(a2-a1))))/(Λb + D_k)
    end 

    Vbz = (Ab/N)*overlap_sum

    return Vbz
end 

function calculate_Va_z_a2(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Aa = 1.703
    Λa = 1.54

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(phi_k-dot(k,a2))+1)/(Λa + D_k)
    end 

    Vaz = (Aa/N)*overlap_sum

    return Vaz
end 

# Heisenberg hopping functions
function calculate_az_az1_overlap(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Aa = 1.704
    Λa = 1.542

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (1+cos(phi_k))*cos(dot(k,a1))/(Λa + D_k)^2 
    end 

    overlap = (2*Aa^2/N)*overlap_sum

    return overlap 
end 

function calculate_bz_bz1_overlap(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.198
    Λb = 0.3009

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (1-cos(phi_k))*cos(dot(k,a1))/(Λb + D_k)^2 
    end 

    overlap = (2*Ab^2/N)*overlap_sum

    return overlap 
end

function calculate_bz1_az_overlap(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.198
    Λb = 0.3009
    Aa = 1.704
    Λa = 1.542

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (sin(phi_k)sin(dot(k,a1)))/((Λb + D_k)*(Λa + D_k))
    end 

    overlap = (2*Ab*Aa/N)*overlap_sum

    return overlap 
end 

function calculate_Ua_z_a1(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Aa = 1.703
    Λa = 1.54

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(phi_k-dot(k,a1))+cos(dot(k,a1)))/(Λa + D_k)
    end 

    Ua1 = (Aa/N)*overlap_sum

    return Ua1
end 

function calculate_Ub_z_a1(BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.198
    Λb = 0.3009

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(dot(k,a1))-cos(phi_k-dot(k,(a1))))/(Λb + D_k)
    end 

    Ub1 = (Ab/N)*overlap_sum

    return Ub1
end 

#=
function heisenberg_hopping()
    a_overlap = 0.0031
    b_overlap = -0.600
    ab_overlap = 0.216
    Ua_0 = 0.849
    Ua_a1 = 0.220
    Ub_0 = 0.443
    Ub_a1 = -0.440
    O = 0.955

    F = 0.52

    term = F*(a_overlap*b_overlap+ab_overlap^2) +
        b_overlap*(Ua_0^2+Ua_a1^2) -
        ab_overlap*(Ub_0*Ua_0+Ua_a1*(-Ub_a1)) -
        ab_overlap*(Ua_0*(-Ub_0)+Ub_a1*Ua_a1) +
        a_overlap*(Ub_0*(-Ub_0)+Ub_a1*(-Ub_a1))

    display(term)

    A_site_hop = cos(O/2)^2+sin(O/2)^2*(b_overlap*a_overlap+ab_overlap^2)

    B_site_hop = (cos(O/2)^2)*(-F) + 
    sin(O)*(Ub_0*Ua_a1+Ua_0*Ub_a1) +
    sin(O/2)^2*term

    display(B_site_hop)

    hop = A_site_hop - B_site_hop

    display(hop)
end 
=#

function magnetic_field_pair_hopping(K=1)
    bb = bb_overlap([0,0],"z",[0,0],"x")
    aa = aa_overlap([0,0],"z",[0,0],"x")
    ab = ab_overlap([0,0],"z",[0,0],"x")

    O = 0.9561
    F0 = 0.5250914141631111

    A_site_hopping = cos(O/2)^2+(sin(O/2)^2)*(bb*aa-ab^2)
    display(A_site_hopping)

    Ua_z_0 = Ua_R_alpha_at_r([0,0],"z",[0,0]) 
    Ub_z_0 = Ub_R_alpha_at_r([0,0],"z",[0,0])
    Ua_z_2 = Ua_R_alpha_at_r([0,0],"z",[0,1])
    Ub_z_2 = Ub_R_alpha_at_r([0,0],"z",[0,1])

    B_site_hopping = -F0*(cos(O/2)^2+(sin(O/2)^2)*(bb*aa-ab^2)) +
                    sin(O)*(Ua_z_0*Ub_z_2 + Ub_z_0*Ua_z_2) +
                    2*(sin(O/2)^2)*(bb*Ua_z_0*Ua_z_2-ab*(Ub_z_0*Ua_z_2-Ua_z_0*Ub_z_2)-aa*Ub_z_0*Ub_z_2)
    display(B_site_hopping)

    hop = A_site_hopping - sign(K)*B_site_hopping
    display(hop)
end

function Interlayer_interaction_pair_hopping()
    bb = bb_overlap([0,0],"z",[0,0],"x")
    aa = aa_overlap([0,0],"z",[0,0],"x")
    ab = ab_overlap([0,0],"z",[0,0],"x")

    O = 0.9561
    F0 = 0.5250914141631111

    A_site_hopping = cos(O/2)^2+(sin(O/2)^2)*(bb*aa-ab^2)
    display(A_site_hopping)

    Ua_z_0 = Ua_R_alpha_at_r([0,0],"z",[0,0]) 
    Ub_z_0 = Ub_R_alpha_at_r([0,0],"z",[0,0])
    Ua_z_2 = Ua_R_alpha_at_r([0,0],"z",[0,1])
    Ub_z_2 = Ub_R_alpha_at_r([0,0],"z",[0,1])

    B_site_hopping = -F0*(cos(O/2)^2+(sin(O/2)^2)*(bb*aa-ab^2)) +
                    sin(O)*(Ua_z_0*Ub_z_2 + Ub_z_0*Ua_z_2) +
                    2*(sin(O/2)^2)*(bb*Ua_z_0*Ua_z_2-ab*(Ub_z_0*Ua_z_2-Ua_z_0*Ub_z_2)-aa*Ub_z_0*Ub_z_2)
    display(B_site_hopping)

    hop = A_site_hopping^2 + B_site_hopping^2
    display(hop)
end

function Gamma_pair_hopping(K=1)
    bb = bb_overlap([1,0],"y",[0,0],"z")
    aa = aa_overlap([1,0],"y",[0,0],"z")
    ab = ab_overlap([1,0],"y",[0,0],"z")

    O = 0.9561
    F0 = 0.5250914141631111

    A_site_hopping = cos(O/2)^2+(sin(O/2)^2)*(bb*aa+ab^2)
    display(A_site_hopping)

    Ua_z_0 = Ua_R_alpha_at_r([0,0],"z",[0,0]) 
    Ub_z_0 = Ub_R_alpha_at_r([0,0],"z",[0,0])
    Ua_z_2 = Ua_R_alpha_at_r([0,0],"z",[1,0])
    Ub_z_2 = Ub_R_alpha_at_r([0,0],"z",[1,0])

    B_site_hopping = -F0*(cos(O/2)^2+(sin(O/2)^2)*(bb*aa+ab^2)) +
                    sin(O)*(Ua_z_0*Ub_z_2 + Ub_z_0*Ua_z_2) +
                    (sin(O/2)^2)*(bb*(Ua_z_0^2+Ua_z_2^2)-2*ab*(Ub_z_0*Ua_z_0-Ua_z_2*Ub_z_2)-aa*(Ub_z_0^2+Ub_z_2^2))
    display(B_site_hopping)

    hop =  sign(K)*B_site_hopping-A_site_hopping
    display(hop)
end

function Heisenberg_pair_hopping(K=1)
    bb = bb_overlap([1,0],"z",[0,0],"z")
    aa = aa_overlap([1,0],"z",[0,0],"z")
    ab = ab_overlap([1,0],"z",[0,0],"z")

    O = 0.9561
    F0 = 0.5250914141631111

    A_site_hopping = cos(O/2)^2+(sin(O/2)^2)*(bb*aa+ab^2)
    display(A_site_hopping)

    Ua_z_0 = Ua_R_alpha_at_r([0,0],"z",[0,0]) 
    Ub_z_0 = Ub_R_alpha_at_r([0,0],"z",[0,0])
    Ua_z_2 = Ua_R_alpha_at_r([0,0],"z",[1,0])
    Ub_z_2 = Ub_R_alpha_at_r([0,0],"z",[1,0])

    B_site_hopping = -F0*(cos(O/2)^2+(sin(O/2)^2)*(bb*aa+ab^2)) +
                    sin(O)*(Ua_z_0*Ub_z_2 + Ub_z_0*Ua_z_2) +
                    (sin(O/2)^2)*(bb*(Ua_z_0^2+Ua_z_2^2)-2*ab*(Ub_z_0*Ua_z_0-Ua_z_2*Ub_z_2)-aa*(Ub_z_0^2+Ub_z_2^2))
    display(B_site_hopping)

    hop =  (A_site_hopping + sign(K)*B_site_hopping)
    display(hop)
end

function Heisenberg_pair_hopping_exact()
    L1 = 25
    L2 = 25
    BCs = [L1,L2,0]
    
    M0 = get_M0(BCs)

    M1 = flip_bond_variable(M0,BCs,[0,0],"z")
    M2 = flip_bond_variable(M0,BCs,[1,0],"z")

    U1 , V1 = get_U_and_V(M1)
    U2 , V2 = get_U_and_V(M2)

    U = U2'*U1
    V = V2'*V1

    X = 0.5*(U+V)
    Y = 0.5*(U-V)

    Z = inv(X)*Y

    C = abs(det(X))^(0.5)
    A_site_hopping = C
    B_site_hopping = C*((U1*V1')[2,1]-(U1*Z*V1')[2,1])
    display(A_site_hopping)
    display(B_site_hopping)

    hop = A_site_hopping - B_site_hopping

    # This second section is for checking alternative formulae 
    U0 ,V0 = get_U_and_V(M0)
    F = (U0'*U1)*(V0'*V1)'

    P_a1 = translate_matrix(BCs,[1,0])
    T_a1 = U0'*P_a1*U0

    A = 0.5*(T_a1*F'+F'*T_a1)
    A_site_hopping_2 = det(A)^(0.5)
    display(A_site_hopping_2)

    B_site_hopping_2 = A_site_hopping_2*(U0*inv(A)*V0')[1,1]
    display(B_site_hopping_2)
    display(hop)
end

function Heisenberg_hopping_M(BCs,J,Q,F)
    M0 = get_M0(BCs)
    U0 ,V0 = get_U_and_V(M0)
    
    Q = [0,0]
    K = 1 

    Δ = diagm(svd(M0).S)

    P_a1 = translate_matrix(BCs,[1,0])
    T_a1 = U0'*P_a1*U0

    P_a2 = translate_matrix(BCs,[0,1])
    T_a2 = U0'*P_a2*U0

    A_a1 = 0.5*(T_a1*F'+F'*T_a1)
    A_a2 = 0.5*(T_a2*F'+F'*T_a2)

    F_tilde = abs(K-J)*F + J*cos(dot(Q,a1))*det(A_a1)^(0.5)*(U0'*V0-sign(K)*inv(A_a1)) + J*cos(dot(Q,a2))*det(A_a2)^(0.5)*(U0'*V0-sign(K)*inv(A_a2))

    display(F_tilde*F')

    M_tilde = abs(K-J)*Δ -2*(U0*F_tilde*F')[1,:]*V0[1,:]'

    return M_tilde
end 

function magnetic_field_pair_hopping_exact()
    L1 = 43
    L2 = 43
    BCs = [L1,L2,0]
    
    M0 = get_M0(BCs)

    M1 = flip_bond_variable(M0,BCs,[0,0],"x")
    M2 = flip_bond_variable(M0,BCs,[0,0],"y")

    U1 , V1 = get_U_and_V(M1)
    U2 , V2 = get_U_and_V(M2)

    U = U2'*U1
    V = V2'*V1

    X = 0.5*(U+V)
    Y = 0.5*(U-V)

    Z = inv(X)*Y

    C = abs(det(X))^(0.5)
    A_site_hopping = C
    B_site_hopping = C*((U1*V1')[1,1]-(U1*Z*V1')[1,1])
    display(A_site_hopping)
    display(B_site_hopping)

    hop = A_site_hopping - B_site_hopping
    display(hop)
end

# Energy density calculation
function Ua_r(r)
    BCs = [200,200,0]

    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Aa = 1.7028
    Λa = 1.5396

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(phi_k-dot(k,r))+cos(dot(k,r)))/(Λa + D_k)
    end 

    Ua_r = (Aa/N)*overlap_sum

    return Ua_r
end 

function Ub_r(r)
    BCs = [200,200,0]
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.1964
    Λb = 0.2995

    overlap_sum = 0 

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(dot(k,r))-cos(phi_k-dot(k,r)))/(Λb + D_k)
    end 

    Ub_r = (Ab/N)*overlap_sum

    return Ub_r
end 

function plot_z_link_energy_density(L_max)
    L1 = L_max
    L2 = L_max
    N = (2*L1+1)*(2*L2+1)
    real_lattice = [n1*a1+n2*a2 for n1 = -L1:L1 for n2 = -L2:L2]

    rx_points = zeros(N)
    ry_points = zeros(N)
    z_link_energy = zeros(N)
    x_link_energy = zeros(N)

    O = 0.955

    cmap = get_cmap("seismic")

    for (id,r) in enumerate(real_lattice)
        rx_points[id] = r[1]
        ry_points[id] = r[2]
        z_link_energy[id] = (cos(O)-1)*(Ua_r(r)*Ua_r(-r)-Ub_r(r)*Ub_r(-r))-sin(O)*(Ua_r(r)*Ub_r(-r)+Ua_r(-r)*Ub_r(r))
        x_link_energy[id] = (cos(O)-1)*(Ua_r(r)*Ua_r(a1-r)-Ub_r(r)*Ub_r(a1-r))-sin(O)*(Ua_r(r)*Ub_r(a1-r)+Ua_r(a1-r)*Ub_r(r))
    end 

    #scatter3D(rx_points,ry_points,ar_points,color = cmap((8*ar_points./maximum(2*ar_points)).+0.5))
    scatter(rx_points,ry_points,color = cmap((z_link_energy./maximum(2*z_link_energy)).+0.5))
    scatter(rx_points,ry_points,color = cmap((x_link_energy./maximum(2*x_link_energy)).+0.5))

    return z_link_energy
end  

function plot_approx_link_energies_2D(L_max)
    N = (2*L_max+1)^2

    A_sites = [n1*a1+n2*a2 for n2=-L_max:(L_max) for n1 = -L_max:(L_max)]

    z_links = [(r,r+rz) for r in A_sites]
    x_links = [(r,r+rx) for r in A_sites]
    y_links = [(r,r+ry) for r in A_sites]

    O = 0.955

    link_energy_of_fluxless_system = 0.5250914141631111

    x_link_energies = zeros(N)
    y_link_energies = zeros(N)
    z_link_energies = zeros(N)

    for (id,r) in enumerate(A_sites)
        z_link_energies[id] = -(cos(O)-1)*(Ua_r(r)*Ua_r(-r)-Ub_r(r)*Ub_r(-r))+sin(O)*(Ua_r(r)*Ub_r(-r)+Ua_r(-r)*Ub_r(r))
        x_link_energies[id] = -(cos(O)-1)*(Ua_r(r)*Ua_r(a1-r)-Ub_r(r)*Ub_r(a1-r))+sin(O)*(Ua_r(r)*Ub_r(a1-r)+Ua_r(a1-r)*Ub_r(r))
        y_link_energies[id] = -(cos(O)-1)*(Ua_r(r)*Ua_r(a2-r)-Ub_r(r)*Ub_r(a2-r))+sin(O)*(Ua_r(r)*Ub_r(a2-r)+Ua_r(a2-r)*Ub_r(r))
    end 

    z_link_energies[Int(ceil(N/2))] =2*link_energy_of_fluxless_system + (cos(O)-1)*(Ua_r([0,0])^2-Ub_r([0,0])^2)-2*sin(O)*(Ua_r([0,0])*Ub_r([0,0]))

    cmap = get_cmap("seismic") # Should choose a diverging colour map so that 0.5 maps to white 

    # max_energy for approximation ~0.215|K|
    # max energy for exact ~0.205
    max_energy = maximum(maximum.([abs.(x_link_energies),abs.(y_link_energies),abs.(z_link_energies)]))
    display(max_energy)

    xcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in x_link_energies]
    ycolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in y_link_energies]
    zcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in z_link_energies]

    zlinecollection = matplotlib.collections.LineCollection(z_links,colors = zcolors,linewidths=2.5)
    xlinecollection = matplotlib.collections.LineCollection(x_links,colors = xcolors,linewidths=2.5)
    ylinecollection = matplotlib.collections.LineCollection(y_links,colors = ycolors,linewidths=2.5)

    display(z_link_energies)

    fig ,ax = subplots()
    #img = imshow([[-max_energy,max_energy]],cmap)
    #colorbar(orientation="horizontal",label = "Energy/K")
    #img.set_visible("False")

    ax.add_collection(zlinecollection)
    ax.add_collection(xlinecollection)
    ax.add_collection(ylinecollection)
    ax.autoscale()
    ax.set_axis_off()
end 

function get_approx_link_energy_around_vison(n_max)
    """
    Calculates the change in link energies in the [-n_max:n_max,-n_max:n_max] region around the vison pair 
    """

    A_sites = [n1*a1+n2*a2 for n2=-n_max:n_max for n1 = -n_max:n_max]

    N = (2*n_max+1)^2

    x_link_energies_approx = zeros(N)
    y_link_energies_approx = zeros(N)
    z_link_energies_approx = zeros(N)

    Θ = 0.9561 # rotation angle from fluxless GS

    for (id,r) in enumerate(A_sites)
        Uar_plus = Ua_r(r)
        Uar_minus = Ua_r(-r)
        Ubr_plus = Ub_r(r)
        Ubr_minus = Ub_r(-r)
        Uar_minus_x = Ua_r(a1-r)
        Uar_minus_y = Ua_r(a2-r)
        Ubr_minus_x = Ub_r(a1-r)
        Ubr_minus_y = Ub_r(a2-r)

        z_link_energies_approx[id] = -(cos(Θ)-1)*(Uar_plus*Uar_minus-Ubr_plus*Ubr_minus)+sin(Θ)*(Uar_plus*Ubr_minus+Uar_minus*Ubr_plus)
        x_link_energies_approx[id] = -(cos(Θ)-1)*(Uar_plus*Uar_minus_x-Ubr_plus*Ubr_minus_x)+sin(Θ)*(Uar_plus*Ubr_minus_x+Uar_minus_x*Ubr_plus)
        y_link_energies_approx[id] = -(cos(Θ)-1)*(Uar_plus*Uar_minus_y -Ubr_plus*Ubr_minus_y)+sin(Θ)*(Uar_plus*Ubr_minus_y+Uar_minus_y *Ubr_plus)
    end 

    F0 = 0.5250914141631111 # Link energy of fluxless system

    z_link_energies_approx[Int(ceil(N/2))] =2*F0 + (cos(Θ)-1)*(Ua_r([0,0])^2-Ub_r([0,0])^2)-2*sin(Θ)*(Ua_r([0,0])*Ub_r([0,0]))
    
    return x_link_energies_approx , y_link_energies_approx , z_link_energies_approx
end 

function plot_link_energies_difference(BCs,n_max,plot_type="diff")
    L1 = BCs[1]
    L2 = BCs[2]

    vison_site = Int.(floor.([L1/2,L2/2]))

    M0 = get_M0(BCs)
    M = flip_bond_variable(M0,BCs,vison_site,"z")

    U0,V0 = get_U_and_V(M0)
    F0 = U0*V0'

    U,V = get_U_and_V(M)
    F = U*V'

    Energy_matrix = -M.*F

    Center_id = convert_n1_n2_to_site_index(vison_site,BCs)
    Central_ids = [(Center_id+n2*L1)+n1 for n2 = -n_max:n_max for n1 = -n_max:n_max]
    A_sites = [n1*a1+n2*a2 for n2 = -n_max:n_max for n1 = -n_max:n_max]

    z_links = [(r,r+rz) for r in A_sites]
    x_links = [(r,r+rx) for r in A_sites]
    y_links = [(r,r+ry) for r in A_sites]

    N_plot = (2*n_max+1)^2
    
    x_link_energy_approx , y_link_energy_approx , z_link_energy_approx = get_approx_link_energy_around_vison(n_max)

    E0 = -F0[1,1]
    x_link_energy_exact = zeros(N_plot)
    y_link_energy_exact = zeros(N_plot)
    z_link_energy_exact = zeros(N_plot)

    for (plot_id,id) in enumerate(Central_ids)
        display
        x_link_energy_exact[plot_id] = Energy_matrix[id,id-1] - E0
        y_link_energy_exact[plot_id] = Energy_matrix[id,id-L1] - E0
        z_link_energy_exact[plot_id] = Energy_matrix[id,id] - E0 
    end 

    cmap = get_cmap("seismic")

    if plot_type == "exact"
        max_energy = maximum(maximum.([abs.(x_link_energy_exact),abs.(y_link_energy_exact),abs.(z_link_energy_exact)]))
        display(max_energy)

        norm = matplotlib.colors.Normalize(-max_energy,max_energy,true)

        xcolors = [cmap(norm(energy)) for energy in x_link_energy_exact]
        ycolors = [cmap(norm(energy)) for energy in y_link_energy_exact]
        zcolors = [cmap(norm(energy)) for energy in z_link_energy_exact]

    elseif plot_type == "approx"
        max_energy = maximum(maximum.([abs.(x_link_energy_approx),abs.(y_link_energy_approx),abs.(z_link_energy_approx)]))
        norm = matplotlib.colors.Normalize(-max_energy,max_energy,true)

        xcolors = [cmap(norm(energy)) for energy in x_link_energy_approx]
        ycolors = [cmap(norm(energy)) for energy in y_link_energy_approx]
        zcolors = [cmap(norm(energy)) for energy in z_link_energy_approx]

    elseif plot_type == "diff"

        x_link_energy_diff = (x_link_energy_exact .- x_link_energy_approx)
        y_link_energy_diff = (y_link_energy_exact .- y_link_energy_approx)
        z_link_energy_diff = (z_link_energy_exact .- z_link_energy_approx)

        max_energy_diff = maximum(maximum.([abs.(x_link_energy_diff),abs.(y_link_energy_diff),abs.(z_link_energy_diff)]))
        #max_energy_diff = 0.22
        norm = matplotlib.colors.Normalize(-max_energy_diff,max_energy_diff,true)
        
        plot((1:N_plot)./N_plot,z_link_energy_exact,color="green")
        plot((1:N_plot)./N_plot,z_link_energy_approx,color="orange")
        plot((1:N_plot)./N_plot,z_link_energy_diff,color = "blue")

        xcolors = [cmap(norm(energy)) for energy in x_link_energy_diff]
        ycolors = [cmap(norm(energy)) for energy in y_link_energy_diff]
        zcolors = [cmap(norm(energy)) for energy in z_link_energy_diff]

    elseif plot_type == "%_diff"
        x_link_energy_diff = (x_link_energy_exact .- x_link_energy_approx)./abs.(x_link_energy_exact)
        y_link_energy_diff = (y_link_energy_exact .- y_link_energy_approx)./abs.(y_link_energy_exact)
        z_link_energy_diff = (z_link_energy_exact .- z_link_energy_approx)./abs.(z_link_energy_exact)

        max_energy = maximum(maximum.([abs.(x_link_energy_diff),abs.(y_link_energy_diff),abs.(z_link_energy_diff)]))
        display(max_energy)

        norm = matplotlib.colors.Normalize(-max_energy,max_energy,true)

        xcolors = [cmap(norm(energy)) for energy in x_link_energy_diff]
        ycolors = [cmap(norm(energy)) for energy in y_link_energy_diff]
        zcolors = [cmap(norm(energy)) for energy in z_link_energy_diff]
    end 


    zlinecollection = matplotlib.collections.LineCollection(z_links,colors = zcolors,linewidths=2.5,cmap = cmap)
    xlinecollection = matplotlib.collections.LineCollection(x_links,colors = xcolors,linewidths=2.5,cmap = cmap)
    ylinecollection = matplotlib.collections.LineCollection(y_links,colors = ycolors,linewidths=2.5,cmap = cmap)

    
    fig ,ax = subplots()
    ax.add_collection(zlinecollection)
    ax.add_collection(xlinecollection)
    ax.add_collection(ylinecollection)
    ax.autoscale()
    ax.set_axis_off()
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap = cmap),location="bottom",label= L"(ε_{r}^{α}-ε_{0})/|K|")
    
end 

function plot_bond_energies_2D(M,BCs)

    F = calculate_F(M)

    A_sites = [n1*a1+n2*a2 for n2=0:(BCs[2]-1) for n1 = 0:(BCs[1]-1)]

    z_links = [(r,r+rz) for r in A_sites]
    x_links = [(r,r+rx) for r in A_sites]
    y_links = [(r,r+ry) for r in A_sites]

    x_link_indices, y_link_indices, z_link_indices = get_link_indices(BCs)

    link_energy_of_fluxless_system = -0.5250914141631111

    x_link_energies = calculate_link_energies(F,M,x_link_indices) .- link_energy_of_fluxless_system
    y_link_energies = calculate_link_energies(F,M,y_link_indices) .- link_energy_of_fluxless_system
    z_link_energies = calculate_link_energies(F,M,z_link_indices) .- link_energy_of_fluxless_system

    cmap = get_cmap("seismic") # Should choose a diverging colour map so that 0.5 maps to white 

    #max_energy = maximum(maximum.([abs.(x_link_energies),abs.(y_link_energies),abs.(z_link_energies)]))
    max_energy = 0.215
    display(max_energy)

    xcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in x_link_energies]
    ycolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in y_link_energies]
    zcolors = [cmap((energy+max_energy)/(2*max_energy)) for energy in z_link_energies]

    zlinecollection = matplotlib.collections.LineCollection(z_links,colors = zcolors,linewidths=2.5)
    xlinecollection = matplotlib.collections.LineCollection(x_links,colors = xcolors,linewidths=2.5)
    ylinecollection = matplotlib.collections.LineCollection(y_links,colors = ycolors,linewidths=2.5)

    fig ,ax = subplots()
    #img = imshow([[-max_energy,max_energy]],cmap)
    #colorbar(orientation="horizontal",label = "Energy/K")
    #img.set_visible("False")

    ax.add_collection(zlinecollection)
    ax.add_collection(xlinecollection)
    ax.add_collection(ylinecollection)
    ax.autoscale()
    ax.set_axis_off()
end 

# General U and V sums 

function Ua_R_alpha_at_r(vison_site_R,α,r_vec)
    BCs = [1000,1000,0]

    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    if α=="x" 
        r_α = rx
    elseif α =="y"
        r_α = ry
    elseif α == "z"
        r_α = rz
    end 

    Aa = 1.7028
    Λa = 1.5396

    overlap_sum = 0 

    R = vison_site_R[1]*a1+vison_site_R[2]*a2
    r = r_vec[1]*a1+r_vec[2]*a2

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(phi_k+dot(k,R-r+r_α-rz))+cos(dot(k,r-R)))/(Λa + D_k)
    end 

    Ua_r = (Aa/N)*overlap_sum

    return Ua_r
end 

function Ub_R_alpha_at_r(vison_site_R,α,r_vec)
    BCs = [1000,1000,0]

    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    if α=="x" 
        r_α = rx
    elseif α =="y"
        r_α = ry
    elseif α == "z"
        r_α = rz
    end 

    Ab = 1.1964
    Λb = 0.2995

    overlap_sum = 0 

    R = vison_site_R[1]*a1+vison_site_R[2]*a2
    r = r_vec[1]*a1+r_vec[2]*a2

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(dot(k,r-R))-cos(phi_k+dot(k,R-r+r_α-rz)))/(Λb + D_k)
    end 

    Ua_r = (Ab/N)*overlap_sum

    return Ua_r
end 

function Va_R_alpha_at_r(vison_site_R,α,r_vec)
    BCs = [100,100,0]

    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    if α=="x" 
        r_α = rx
    elseif α =="y"
        r_α = ry
    elseif α == "z"
        r_α = rz
    end 

    Aa = 1.703
    Λa = 1.54

    overlap_sum = 0 

    R = vison_site_R[1]*a1+vison_site_R[2]*a2
    r = r_vec[1]*a1+r_vec[2]*a2

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(dot(k,R-r+r_α-rz))+cos(dot(k,r-R)+phi_k))/(Λa + D_k)
    end 

    Va_r = (Aa/N)*overlap_sum

    return Va_r
end 

function aa_overlap(R_vec,α,R_prime_vec,β)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Aa = 1.7028
    Λa = 1.5396

    if α=="x" 
        r_α = rx
    elseif α =="y"
        r_α = ry
    elseif α == "z"
        r_α = rz
    end 

    if β=="x" 
        r_β = rx
    elseif β =="y"
        r_β = ry
    elseif β == "z"
        r_β = rz
    end 

    overlap_sum = 0 

    R = R_vec[1]*a1 + R_vec[2]*a2
    R_prime = R_prime_vec[1]*a1 + R_prime_vec[2]*a2

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(dot(k,R-R_prime))+cos(dot(k,R-R_prime+r_α-rz)+phi_k)+cos(dot(k,R-R_prime-r_β+rz)-phi_k)+cos(dot(k,R-R_prime+r_α-r_β)))/(Λa + D_k)^2 
    end 

    overlap = (Aa^2/N)*overlap_sum

    return overlap 
end 

function bb_overlap(R_vec,α,R_prime_vec,β)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.1964
    Λb = 0.2995

    if α=="x" 
        r_α = rx
    elseif α =="y"
        r_α = ry
    elseif α == "z"
        r_α = rz
    end 

    if β=="x" 
        r_β = rx
    elseif β =="y"
        r_β = ry
    elseif β == "z"
        r_β = rz
    end 

    overlap_sum = 0 

    R = R_vec[1]*a1 + R_vec[2]*a2
    R_prime = R_prime_vec[1]*a1 + R_prime_vec[2]*a2

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(dot(k,R-R_prime))-cos(dot(k,R-R_prime+r_α-rz)+phi_k)-cos(dot(k,R-R_prime-r_β+rz)-phi_k)+cos(dot(k,R-R_prime+r_α-r_β)))/(Λb + D_k)^2 
    end 

    overlap = (Ab^2/N)*overlap_sum

    return overlap 
end 

function ab_overlap(R_vec,α,R_prime_vec,β)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Ab = 1.1964
    Λb = 0.2995
    Aa = 1.7028
    Λa = 1.5396

    if α=="x" 
        r_α = rx
    elseif α =="y"
        r_α = ry
    elseif α == "z"
        r_α = rz
    end 

    if β=="x" 
        r_β = rx
    elseif β =="y"
        r_β = ry
    elseif β == "z"
        r_β = rz
    end 

    overlap_sum = 0 

    R = R_vec[1]*a1 + R_vec[2]*a2
    R_prime = R_prime_vec[1]*a1 + R_prime_vec[2]*a2

    for (id,k) in enumerate(k_lattice)
        D_k ,cos_phi_k = Delta_k(k)
        phi_k = angle(1+exp(im*dot(k,a1))+exp(im*dot(k,a2)))
        overlap_sum += (cos(dot(k,R-R_prime))+cos(dot(k,R-R_prime+r_α-rz)+phi_k)-cos(dot(k,R-R_prime-r_β+rz)-phi_k)-cos(dot(k,R-R_prime+r_α-r_β)))/((Λb + D_k)*(Λa+D_k)) 
    end 

    overlap = (Ab*Aa/N)*overlap_sum

    return overlap 
end 

#= Older Version of calculation

display(2*Fv[1,1]-2*F0[1,1]) # exact <2|V|2> - <1|V|1>
display(-2sin(O)*(a[1]*(F0*b)[1]-b[1]*(F0*a)[1])-4*sin(O/2)^2*(a[1]*(F0*a)[1]+b[1]*(F0*b)[1])) # estimate <2|V|2> - <1|V|1>
display(-2*im*sin(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])-2*(1-cos(O))*(z[1]*(F0*conj(z))[1]+conj(z)[1]*(F0*z)[1]))

display(tr(M0*(F0'-Fv'))) #exact {0^2}H^1{0^2}
display(2*sin(O/2)^2*(b'*(M0'*F0)*b+a'*(M0'*F0)*a)) #estimate in terms of a and b 
display(4*sin(O/2)^2*z'*(M0'*F0)*z) # variational estimate in terms of complex z
#display(-2*sin(O/2)^2*(Lambda_a-2*Aa*a[1]+Lambda_b-2*Ab*b[1]))

display(tr(Mv*Fv')-tr(M0*F0')-F0[1,1])
display((cos(O)-1)*(2*(a[1])^2-2*(b[1])^2-(b'*(M0'*F0)*b+a'*(M0'*F0)*a))+4*sin(O)*(a[1])*b[1])
#display((cos(O)-1)*(2*(a[1])^2-2*(b[1])^2+Lambda_a-2*Aa*a[1]+Lambda_b-2*Ab*b[1])+4*sin(O)*(a[1])*b[1])

#display(Lambda_a + a'*(M0'*F0)*a -2*Aa*a[1])
#display(Lambda_b + b'*(M0'*F0)*b -2*Ab*b[1])

display((a[1]-Aa)*tan(O/2)-b[1]) # Equation from minimising E with respect to Λa
display((b[1]+Ab)*tan(O/2)+a[1]) # Equation from minimising E with respect to Λb
display(-a[1]^2+a[1]*Aa-b[1]^2-b[1]*Ab)



Λ1 = (cos(O)-1)*(z'*M0'*F0*z-2*real(conj(z[1])*(F0*z)[1]))+2*sin(O)*(imag(conj(z)[1]*(F0*z)[1]))
Λ2 = (cos(O)-1)*(transpose(z)*M0'*F0*z-2*z[1]*(F0*z)[1])
Eq = Λ2*conj(z)+Λ1*z -(cos(O)-1)*(M0'*F0*z-z[1]*F0[1,:])-im*sin(O)*(z[1]*F0[1,:])

Λ1_tilde = sin(O/2)*(transpose(z)*M0'*F0*z-2*z[1]*(F0*z)[1])
Λ2_tilde = sin(O/2)*(z'*M0'*F0*z) - 2*imag(z'[1]*(F0*z)[1]*exp(im*O/2))
Eq_tilde = Λ1_tilde*conj(z)+Λ2_tilde*z - sin(O/2)*M0'*F0*z +im*(z[1]*F0[1,:]*exp(-im*O/2))

xi_plus = (conj(F0*z)[1]*z+z[1]*conj(z))./(sqrt(2)*abs(z[1]))
xi_minus = (conj(F0*z)[1]*z-z[1]*conj(z))./(sqrt(2)*abs(z[1]))

Eq_plus = (abs(Λ1_tilde)+Λ2_tilde)xi_plus-sin(O/2)*M0'*F0*xi_plus -im*abs(z[1])*exp(im*O/2)*(1-exp(im*(phi-O)))*(1/sqrt(2))*F0[1,:]
Eq_minus = (-abs(Λ1_tilde)+Λ2_tilde)xi_minus-sin(O/2)*M0'*F0*xi_minus +im*abs(z[1])*exp(im*O/2)*(1+exp(im*(phi-O)))*(1/sqrt(2))*F0[1,:]

a = real.(xi_plus*exp(-im*phi/2))
b = real.(xi_minus*(-im*exp(-im*phi/2)))

Lambda_a = (Λ2_tilde + abs(Λ1_tilde))/sin(-O/2)
Lambda_b = (Λ2_tilde - abs(Λ1_tilde))/sin(-O/2) 

Aa = sqrt(2)*abs(z[1])*sin((phi-O)/2)/sin(-O/2) 
Ab = sqrt(2)*abs(z[1])*cos((phi-O)/2)/sin(-O/2) 

Eq_a = Lambda_a*a+M0'*F0*a - Aa*F0[1,:]
Eq_b = Lambda_b*b+M0'*F0*b + Ab*F0[1,:]


Λ = (cos(O)-1)*(2*b'*M0'*F0*a-2(b[1]*(F0*a)[1]+a[1]*(F0*b)[1]))
Λa = 0.5*(cos(O)-1)*(2*a'*M0'*F0*a-4*a[1]*(F0*a)[1])-sin(O)*(b[1]*(F0*a)[1]-a[1]*(F0*b)[1])
Λb = 0.5*(cos(O)-1)*(2*b'*M0'*F0*b-4*b[1]*(F0*b)[1])-sin(O)*(b[1]*(F0*a)[1]-a[1]*(F0*b)[1])

Eqa = (1-cos(O))*(2*M0'*F0*a-2*a[1]*F0[1,:]) + 2*sin(O)*(b[1]*F0[1,:]) + Λ*b + 2*Λa*a
Eqb = (1-cos(O))*(2*M0'*F0*b-2*b[1]*F0[1,:]) - 2*sin(O)*(a[1]*F0[1,:]) + Λ*a + 2*Λb*b 

ak = V0'*a
bk = V0'*b

Ez = 4*sin(O)*(U0*bk)[1]*(U0*ak)[1]+2*(cos(O)-1)*((U0*ak)[1]^2-(U0*bk)[1]^2-a'*M0'*F0*a-b'*M0'*F0*b)

#=
function vary_z_phase(z0)
    for θ in LinRange(0,pi,100)
        z = z0*exp(im*θ)
        Λ2 = (cos(O)-1)*(transpose(z)*M0'*F0*z-2*z[1]*(F0*z)[1])
        scatter(θ,angle(Λ2),color="b")
        scatter(θ,angle(z[1]),color="r")
        scatter(θ,angle(Λ2*conj(z)[1]))
        #=
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
        =#
    end 
end 
=#

function plot_variational_params_vs_BCs(L_max)
    for L = 10:L_max
        BCs = [L,L,0]
        M0 = get_M0(BCs)
        Mv = flip_bond_variable(M0,BCs,[0,0],"z")

        U0, V0 = get_U_and_V(M0)
        Uv ,Vv = get_U_and_V(Mv)
        F0 = U0*V0'

        F = (U0'*Uv)*(V0'*Vv)'

        O = angle(eigvals(F)[1])
        Q = eigvecs(F)
        z = V0*Q[:,1]

        xi_plus = (conj(F0*z)[1]*z+z[1]*conj(z))./(sqrt(2)*abs(z[1]))
        xi_minus = (conj(F0*z)[1]*z-z[1]*conj(z))./(sqrt(2)*abs(z[1]))

        phi = angle(z[1]/(F0*z)[1])
        display(phi)
        display(O)

        a = real.(xi_plus*exp(-im*phi/2))
        b = real.(xi_minus*(-im*exp(-im*phi/2)))

        Λ1_tilde = sin(O/2)*(transpose(z)*M0'*F0*z-2*z[1]*(F0*z)[1])
        Λ2_tilde = sin(O/2)*(z'*M0'*F0*z) - 2*imag(z'[1]*(F0*z)[1]*exp(im*O/2))

        Lambda_a = real((Λ2_tilde + abs(Λ1_tilde)))/sin(-O/2)
        Lambda_b = real(Λ2_tilde - abs(Λ1_tilde))/sin(-O/2) 

        Aa = sqrt(2)*abs(z[1])*sin((phi-O)/2)/sin(-O/2)
        Ab = sqrt(2)*abs(z[1])*cos((phi-O)/2)/sin(-O/2)

        #scatter(1/L,Lambda_a,color="b")
        #scatter(1/L,a[1],color="b",marker="x")
        #scatter(1/L,a[1],color="b",marker="o")
        #scatter(1/L,b[1],color="r",marker="x")
        #scatter(1/L,Lambda_b,color="r")
        #scatter(1/L,b[1],color="r",marker="o")
        scatter(1/L,Lambda_a-Lambda_b,color="b")
        scatter(1/L,Lambda_a+Lambda_b,color="r")
    end 
end 

#E2 = -2*im*sin(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])+2*(1-cos(O))*(z'*(M0'*F0)*z-z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])

#dE2_dO = -2*im*cos(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])+2*(sin(O))*(z'*(M0'*F0)*z-z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])

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

#=
function Lambda_sums(Λ,BCs)
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    Sum_term = 0 
    cos_term = 0
    square_term = 0 
    cos_square_term = 0 
    energy_term = 0
    energy_cos_term = 0 
    for k in k_lattice
        D_k ,cos_phi_k = Delta_k(k)
        cos_term += (cos_phi_k)/(Λ + D_k)
        Sum_term += 1/(Λ + D_k)
        square_term += 1/(Λ + D_k)^2
        cos_square_term += cos_phi_k/(Λ + D_k)^2
        energy_term += D_k/(Λ+D_k)^2
        energy_cos_term += D_k*cos_phi_k/(Λ+D_k)^2
    end 
    return (2/N)*Sum_term, (2/N)*cos_term, (2/N)*square_term , (2/N)*cos_square_term , (2/N)*energy_term, (2/N)*energy_cos_term
end 

a_sum , a_cos_sum , a_square_sum ,a_cos_square_sum , a_energy_sum , a_energy_cos_sum = Lambda_sums(1.53,BCs)

E_a = a_energy_sum + a_energy_cos_sum
=#
=#