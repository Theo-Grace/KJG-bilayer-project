# These are numerical calculations used to calculate the Matter sector ground state for the case of open flux pairs 
# The functions used come from Bilayer_Variationalcalc_2.jl 

# Sets the boundary conditions 
L1 = 28
L2 = 28
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

Q = eigvecs(F)
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
Eq_a = Λa*a + M0'*F0*a + Aa*F0[1,:]
Eq_b = Λb*b + M0'*F0*b + Ab*F0[1,:]
display((a[1]-Aa)*tan(O/2)-b[1]) # Equation from minimising E with respect to Λa
display((b[1]+Ab)*tan(O/2)+a[1]) # Equation from minimising E with respect to Λb
display(-a[1]^2+a[1]*Aa-b[1]^2-b[1]*Ab)


# Variational calculation

function get_Aa(Λa)
    BCs = [100,100,0]
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
    BCs = [100,100,0]
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
    BCs= [100,100,0]
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
    BCs = [100,100,0]
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

function find_Lambda_b_from_Lambda_a(Λa)
    Ua = get_Ua(Λa)
    Aa = get_Aa(Λa)

    func_a = Ua*Aa-Ua^2
    func_b = 0 
    Λb = 0 

    while func_a - func_b > 0 
        Ub = get_Ub(Λb)
        Ab = get_Ab(Λb)
        func_b = Ub^2+Ub*Ab
        Λb +=0.001
        #display(Λb)
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
        #scatter(Λ,O_point)
    
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





#= Older Version of calculation

display(2*Fv[1,1]-2*F0[1,1]) # exact <2|V|2> - <1|V|1>
display(-2sin(O)*(a[1]*(F0*b)[1]-b[1]*(F0*a)[1])-4*sin(O/2)^2*(a[1]*(F0*a)[1]+b[1]*(F0*b)[1])) # estimate <2|V|2> - <1|V|1>
display(-2*im*sin(O)*(z[1]*(F0*conj(z))[1]-conj(z)[1]*(F0*z)[1])-2*(1-cos(O))*(z[1]*(F0*conj(z))[1]+conj(z)[1]*(F0*z)[1]))

display(tr(M0*(F0'-Fv'))) #exact {0^2}H^1{0^2}
display(2*sin(O/2)^2*(b'*(M0'*F0)*b+a'*(M0'*F0)*a)) #estimate in terms of a and b 
display(4*sin(O/2)^2*z'*(M0'*F0)*z) # variational estimate in terms of complex z
display(-2*sin(O/2)^2*(Lambda_a-2*Aa*a[1]+Lambda_b-2*Ab*b[1]))

display(tr(Mv*Fv')-tr(M0*F0')-F0[1,1])
display((cos(O)-1)*(2*(a[1])^2-2*(b[1])^2-(b'*(M0'*F0)*b+a'*(M0'*F0)*a))+4*sin(O)*(a[1])*b[1])
display((cos(O)-1)*(2*(a[1])^2-2*(b[1])^2+Lambda_a-2*Aa*a[1]+Lambda_b-2*Ab*b[1])+4*sin(O)*(a[1])*b[1])

display(Lambda_a + a'*(M0'*F0)*a -2*Aa*a[1])
display(Lambda_b + b'*(M0'*F0)*b -2*Ab*b[1])

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

Eq_a = Lambda_a*a+M0'*F0*a -Aa*F0[1,:]


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