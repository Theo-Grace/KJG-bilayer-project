# This is a numerical calculation of the Green's functions for Majoranas in Kitaev's model 

using LinearAlgebra
using SparseArrays
using Arpack 
using PyPlot
pygui(true) 
using QuadGK
using HDF5

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

g1,g2 = dual(a1,a2)
# There are two distinct elements in the 2x2 Majorana Greens function
# G^AA = G^BB 
# G^AB = -G^BA 
# 1D Integral expressions can be found for IM(G^AA) and Re(G^AB)

# This section is for the imaginary part of G^AA

function ImG0AA_at_w(w,tol=1e-7)
    AA_Integrand(theta) = -(w/pi)*(((w^2+4)/8)*cos(theta)^2 - ((w^2-4)/16)^2 -cos(theta)^4)^(-1/2)
    if 2<w<6
        ImG0AA,est = quadgk(AA_Integrand,0,acos((w-2)/4),rtol=tol)
    elseif 0<w<2
        ImG0AA,est = quadgk(AA_Integrand,acos((2+w)/4),acos((2-w)/4),rtol=tol)
    end
    return ImG0AA
end 

function get_ImG0AA(Num_points=1000)
    ImG0AA = zeros(2*Num_points)
    for (id,w) = enumerate(LinRange(0.15,5.9999,Num_points))
        ImG0AA[Num_points+id] = ImG0AA_at_w(w)
        ImG0AA[Num_points+1-id] = -ImG0AA_at_w(w)
        display(id)
    end
    w_points = [-reverse(collect(LinRange(0.15,5.99,Num_points)));collect(LinRange(0.15,5.99,Num_points))]
    return w_points, ImG0AA
end 

function Interpolate_ImG0AA(w)
    if abs(w) > 5.97
        Interpolated_ImG0AA = 0
    else
        b=findfirst((x -> x>w),ImG0AA_w_points)
        a=b-1
        Interpolated_ImG0AA = ImG0AA[a]+(w-ImG0AA_w_points[a])*(ImG0AA[b]-ImG0AA[a])/(ImG0AA_w_points[b]-ImG0AA_w_points[a])
    end
    return Interpolated_ImG0AA
end 

function ReG0AA_at_w(w,tol=1e-7)
    eta = 1e-6
    Integrand(w_prime) = (1/pi)*Interpolate_ImG0AA(w_prime)*((w_prime-w)/((w_prime-w)^2+eta^2))
    ReG0AA,est = quadgk(Integrand,-5.99,5.99,rtol=tol)
    return ReG0AA
end

function get_ReG0AA(Num_points=1000)
    ReG0AA = zeros(2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,100,Num_points))
        ReG0AA[Num_points+id] = ReG0AA_at_w(w)
        ReG0AA[Num_points+1-id] = -ReG0AA_at_w(w)
        display(id)
    end
    w_points = [-reverse(collect(LinRange(0.01,100,Num_points)));collect(LinRange(0.01,100,Num_points))]
    return w_points, ReG0AA
end 

function Interpolate_ReG0AA(w)
    b=findfirst((x -> x>w),ReG0AA_w_points)
    a=b-1
    Interpolated_ReG0AA = ReG0AA[a]+(w-ReG0AA_w_points[a])*(ReG0AA[b]-ReG0AA[a])/(ReG0AA_w_points[b]-ReG0AA_w_points[a])
    return Interpolated_ReG0AA
end 

# This section adds functions to calculate G0AB

function ReG0AB_at_w(w,tol=1e-7)
    AA_Integrand(t) = (1/pi)*((w^2+4)/4 - 4/(1+t^2))*(((w^2+4)/8)*(1+t^2) - ((1+t^2)^2)*(((w^2-4)/16)^2) -1)^(-1/2)
    eta = 10^(-10)
    if 2<w<6
        ReG0AB,est = quadgk(AA_Integrand,0,tan(acos((w-2)/4))-eta,rtol=tol)
    elseif 0<w<2
        ReG0AB,est = quadgk(AA_Integrand,tan(acos((2+w)/4))+eta,tan(acos((2-w)/4))-eta,rtol=tol)
    end
    return ReG0AB
end 

function get_ReG0AB(Num_points=1000)
    ReG0AB = zeros(2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,5.9999,Num_points))
        display(id)
        ReG0AB[Num_points+id] = ReG0AB_at_w(w)
        ReG0AB[Num_points+1-id] = ReG0AB_at_w(w)
    end
    w_points = [-reverse(collect(LinRange(0.01,5.99,Num_points)));collect(LinRange(0.01,5.99,Num_points))]
    return w_points, ReG0AB
end 

function Interpolate_ReG0AB(w)
    if abs(w) > 5.97
        Interpolated_ReG0AB = 0
    else
        b = findfirst((x -> x>w),ReG0AB_w_points)
        a=b-1
        Interpolated_ReG0AB = ReG0AB[a]+(w-ReG0AB_w_points[a])*(ReG0AB[b]-ReG0AB[a])/(ReG0AB_w_points[b]-ReG0AB_w_points[a])
    end 
    return Interpolated_ReG0AB
end 

function ImG0AB_at_w(w,tol=1e-7)
    eta = 1e-6
    Integrand(w_prime) = (1/pi)*Interpolate_ReG0AB(w_prime)*((w_prime-w)/((w_prime-w)^2+eta^2))
    ImG0AB_,est = quadgk(Integrand,-5.99,5.99,rtol=tol)
    return ImG0AB_
end

function get_ImG0AB(Num_points=1000)
    ImG0AB_ = zeros(2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,100,Num_points))
        ImG0AB_[Num_points+id] = ImG0AB_at_w(w)
        ImG0AB_[Num_points+1-id] = ImG0AB_at_w(w)
        display(id)
    end
    w_points = [-reverse(collect(LinRange(0.01,100,Num_points)));collect(LinRange(0.01,100,Num_points))]
    return w_points, ImG0AB_
end 

function Interpolate_ImG0AB(w)
    b=findfirst((x -> x>w),ImG0AB_w_points)
    a=b-1
    Interpolated_ImG0AB = ImG0AB[a]+(w-ImG0AB_w_points[a])*(ImG0AB[b]-ImG0AB[a])/(ImG0AB_w_points[b]-ImG0AB_w_points[a])
    return Interpolated_ImG0AB
end 

# This section adds functions to save and load data
# Note that ReG0AA ~< 0.1 for |w|>15 
function save_ImG0AA_data(w_points,ImG0AA,tol=1e-7)
    Num_points = size(w_points)[1]/2

    Description = "The number of points used to calculate ReG0AA was $Num_points 
    The tolerance for numerical integration was tol=$tol.
    The Integration was perfomed using quadgk"

    group_name = "ImG0AA_data"

    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AA","cw")

    create_group(fid,group_name)
    g = fid[group_name]
    write(g,"Description",Description)
    write(g,"w_points",w_points)
    write(g,"Imaginary part of G0AA",ImG0AA)
    close(fid)
end

function load_ImG0AA()
    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AA","r")
    group_name = "ImG0AA_data"
    g = fid[group_name]
    w_points = read(g["w_points"])
    ImG0AA = read(g["Imaginary part of G0AA"])
    close(fid)

    return w_points, ImG0AA
end 

function save_ReG0AA_data(w_points,ReG0AA,tol=1e-7)
    Num_points = size(w_points)[1]/2

    Description = "The number of points used to calculate ReG0AA was $Num_points 
    The tolerance for numerical integration was tol=$tol.
    The Integration was perfomed using quadgk"

    group_name = "ReG0AA_data"

    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AA","cw")

    create_group(fid,group_name)
    g = fid[group_name]
    write(g,"Description",Description)
    write(g,"w_points",w_points)
    write(g,"Real part of G0AA",ReG0AA)
    close(fid)
end

function load_ReG0AA()
    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AA","r")
    group_name = "ReG0AA_data"
    g = fid[group_name]
    w_points = read(g["w_points"])
    ReG0AA = read(g["Real part of G0AA"])
    close(fid)

    return w_points, ReG0AA
end 

function save_ReG0AB_data(w_points,ReG0AB,tol=1e-7)
    Num_points = size(w_points)[1]/2

    Description = "The number of points used to calculate ReG0AB was $Num_points 
    The tolerance for numerical integration was tol=$tol.
    The Integration was perfomed using quadgk"

    group_name = "ReG0AB_data"

    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AB","cw")

    create_group(fid,group_name)
    g = fid[group_name]
    write(g,"Description",Description)
    write(g,"w_points",w_points)
    write(g,"Real part of G0AB",ReG0AB)
    close(fid)
end

function load_ReG0AB()
    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AB","r")
    group_name = "ReG0AB_data"
    g = fid[group_name]
    w_points = read(g["w_points"])
    ReG0AB = read(g["Real part of G0AB"])
    close(fid)

    return w_points, ReG0AB
end 

function save_ImG0AB_data(w_points,ImG0AB,tol=1e-7)
    Num_points = size(w_points)[1]/2

    Description = "The number of points used to calculate ReG0AB was $Num_points 
    The tolerance for numerical integration was tol=$tol.
    The Integration was perfomed using quadgk"

    group_name = "ImG0AB_data"

    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AB","cw")

    create_group(fid,group_name)
    g = fid[group_name]
    write(g,"Description",Description)
    write(g,"w_points",w_points)
    write(g,"Imaginary part of G0AB",ImG0AB)
    close(fid)
end

function load_ImG0AB()
    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AB","r")
    group_name = "ImG0AB_data"
    g = fid[group_name]
    w_points = read(g["w_points"])
    ImG0AB = read(g["Imaginary part of G0AB"])
    close(fid)

    return w_points, ImG0AB
end 

# Use this to load the saved Green's functions

ImG0AA_w_points, ImG0AA = load_ImG0AA()
ReG0AA_w_points, ReG0AA = load_ReG0AA()
ImG0AB_w_points, ImG0AB = load_ImG0AB()
ReG0AB_w_points, ReG0AB = load_ReG0AB()


function G0_at_w(w)
    G0AA = Interpolate_ReG0AA(w)+im*Interpolate_ImG0AA(w)
    G0AB = Interpolate_ReG0AB(w)+im*Interpolate_ImG0AB(w)

    G0 = [G0AA G0AB ; -G0AB G0AA]
    return G0
end 

function get_G0(Num_points=1000)
    G0 = zeros(Complex{Float64},2,2,2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,99,Num_points))
        G0[:,:,Num_points+id] = G0_at_w(w)
        G0[:,:,Num_points+1-id] = G0_at_w(-w)
        display(w)
    end
    w_points = [-reverse(collect(LinRange(0.01,99,Num_points)));collect(LinRange(0.01,99,Num_points))]
    return w_points, G0
end 

function V_at_w(w)
    G0AA = Interpolate_ReG0AA(w)+im*Interpolate_ImG0AA(w)
    G0AB = Interpolate_ReG0AB(w)+im*Interpolate_ImG0AB(w)

    G0 = [G0AA G0AB ; -G0AB G0AA]
    V = 2*inv([0 -im ; im 0] - 2*G0)
    return V
end 

function get_V(Num_points=1000)
    V = zeros(Complex{Float64},2,2,2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,99,Num_points))
        V[:,:,Num_points+id] = V_at_w(w)
        V[:,:,Num_points+1-id] = V_at_w(-w)
        display(w)
    end
    w_points = [-reverse(collect(LinRange(0.01,99,Num_points)));collect(LinRange(0.01,99,Num_points))]
    return w_points, V
end 

function Gf0_at_w(w)
    G0_Majoranas = G0_at_w(w)
    Gf0 = G0_Majoranas[1,1]+im*G0_Majoranas[1,2]

    return Gf0
end 

function plot_Gf0()
    Num_points = 1000
    w_points = LinRange(-5.98,5.98,Num_points)
    Gf0 = zeros(Complex{Float64},Num_points)
    for (id,w) in enumerate(w_points)
        Gf0[id] = Gf0_at_w(w)
    end 

    plot(w_points,real(Gf0))
    plot(w_points,imag(Gf0))
end 

function plot_G0AB()
        Num_points = 1000
        w_points = LinRange(-5.98,5.98,Num_points)
        Gf = zeros(Complex{Float64},Num_points)
        for (id,w) in enumerate(w_points)
            Gf[id] = 1/(1+4Gf0_at_w(w))
            display(w)
        end 

    plot(w_points,real(Gf),color="b")
    plot(w_points,imag(Gf),color="b",linestyle="dotted")
end


# This section is for a calculation based on real time correlators 

function Delta_at_k(k)
    Delta = 1 + exp(im*dot(k,a1)) + exp(im*dot(k,a2))
    return Delta
end

function R_0k(k)
    Delta = Delta_at_k(k)
    abs_Delta = abs(Delta)
    cos_phi = real(Delta)/abs_Delta
    sin_phi = imag(Delta)/abs_Delta

    R = [ 1 0 0 0 ; 1 0 0 0 ; 0 0 cos_phi sin_phi ; 0 0 -sin_phi cos_phi]

    return R 
end 

function gamma_Greens_function(k,w)
    Delta = abs(Delta_at_k(k))
    f_plus = 1/(w+2*Delta) + 1/(w-2*Delta)
    f_minus = 1/(w+2*Delta) - 1/(w-2*Delta)

    G_k = [f_plus*I(2) -f_minus*I(2); f_minus*I(2) f_plus*I(2)]

    return G_k
end 

function C_k_k_prime(k,k_prime)
    w_k_prime = 2*abs(Delta_at_k(k_prime)) 
    C_0 = [-im*I(2) I(2) ; -I(2) -im*I(2)]
    R_k_prime = R_0k(k_prime)
    R_k = R_0k(k)'
    V_2x2 = V_at_w(w_k_prime/2)
    V_4x4 = [V_2x2[1,1]*I(2) V_2x2[1,2]*I(2) ; V_2x2[2,1]*I(2) V_2x2[2,2]*I(2)]
    Gk = gamma_Greens_function(k,w_k_prime)

    display(Gk)

    C = Gk*R_k*V_4x4*R_k_prime*C_0

    return C
end

# This section adds code to calculate the density matrix for f fermions on the flux site 

function Delta_k(k)
    Delta = 1 + exp(im*dot(k,a1)) + exp(im*dot(k,a2))

    cos_phi = real(Delta)/abs(Delta)

    return abs(Delta) , cos_phi
end 

function plot_unit_cell_in_reciprocal_space(BCs)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    kx_points = [k[1] for k in k_lattice]
    ky_points = [k[2] for k in k_lattice]

    scatter(kx_points,ky_points)
end 

function calculate_on_site_fluxless_f_fermion_density(BCs)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    k_lattice = [(j1/L1)*g1 + (j2/L2 -(m*j1)/(L1*L2))*g2 for j1 = 0:(L1-1) for j2 = 0:(L2-1)]
    
    onsite_density = 0 

    for k in k_lattice
        abs_Delta_k , cos_phi_k = Delta_k(k)
        onsite_density += 0.5*(1+cos_phi_k)
    end 

    return onsite_density/(L1*L2)
end

function calculate_on_site_fluxless_F00(BCs)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    k_lattice = [(j1/L1)*g1 + (j2/L2 -(m*j1)/(L1*L2))*g2 for j1 = 0:(L1-1) for j2 = 0:(L2-1)]
    
    onsite_F00 = 0 

    for k in k_lattice
        abs_Delta_k , cos_phi_k = Delta_k(k)
        onsite_F00+= cos_phi_k
    end 

    return onsite_F00/(L1*L2)
end

function ImG0f00_ret_at_w(w,tol=1e-7)
    G0f00_Integrand(theta) = (abs(w)/(8pi))*(1+sign(w)*((w^2+4-16*cos(theta)^2)/(4*abs(w))))*(((w^2+4)/8)*cos(theta)^2 - ((w^2-4)/16)^2 -cos(theta)^4)^(-1/2)
    if 2<abs(w)<6
        ImG0f00,est = quadgk(G0f00_Integrand,0,acos((abs(w)-2)/4),rtol=tol)
    elseif 0<abs(w)<2
        ImG0f00,est = quadgk(G0f00_Integrand,acos((2+abs(w))/4),acos((2-abs(w))/4),rtol=tol)
    end
    return ImG0f00
end 

function plot_ImG0f00_ret()
    Num_points = 1000
    w_points = LinRange(-5.98,5.98,Num_points)
    ImG0f00 = zeros(Complex{Float64},Num_points)
    for (id,w) in enumerate(w_points)
        ImG0f00[id] = ImG0f00_ret_at_w(w,1e-5)
    end 
    plot(w_points,ImG0f00)
end 

function get_ImG0f00_ret(Num_points=1000)
    ImG0f00= zeros(2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,5.9999,Num_points))
        ImG0f00[Num_points+id] = ImG0f00_ret_at_w(w,1e-5)
        ImG0f00[Num_points+1-id] = ImG0f00_ret_at_w(-w,1e-5)
        display(id)
    end
    w_points = [-reverse(collect(LinRange(0.01,5.99,Num_points)));collect(LinRange(0.01,5.99,Num_points))]
    return w_points, ImG0f00
end 

function Interpolate_ImG0f00_ret(w,ImG0f00_w_points=[-reverse(collect(LinRange(0.01,5.99,1000)));collect(LinRange(0.01,5.99,1000))])
    if abs(w) > 5.97
        Interpolated_ImG0f00 = 0
    else
        b=findfirst((x -> x>w),ImG0f00_w_points)
        a=b-1
        Interpolated_ImG0f00 = ImG0f00[a]+(w-ImG0f00_w_points[a])*(ImG0f00[b]-ImG0f00[a])/(ImG0f00_w_points[b]-ImG0f00_w_points[a])
    end
    return Interpolated_ImG0f00
end 

function ReG0f00_ret_at_w(w,tol=1e-7)
    eta = 1e-6
    Integrand(w_prime) = (-1/pi)*Interpolate_ImG0f00_ret(w_prime)*((w_prime-w)/((w_prime-w)^2+eta^2))
    ReG0f00,est = quadgk(Integrand,-5.99,5.99,rtol=tol)
    return ReG0f00
end

function get_ReG0f00_ret(Num_points=1000)
    ReG0f00= zeros(2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,5.9999,Num_points))
        ReG0f00[Num_points+id] = ReG0f00_ret_at_w(w,1e-5)
        ReG0f00[Num_points+1-id] = ReG0f00_ret_at_w(-w,1e-5)
        display(id)
    end
    w_points = [-reverse(collect(LinRange(0.01,5.99,Num_points)));collect(LinRange(0.01,5.99,Num_points))]
    return w_points, ReG0f00
end 


function plot_G0f00_ret()
    Num_points = 1000
    w_points = LinRange(-5.98,5.98,Num_points)
    ImG0f00 = zeros(Complex{Float64},Num_points)
    ReG0f00 = zeros(Complex{Float64},Num_points)
    for (id,w) in enumerate(w_points)
        ImG0f00[id] = ImG0f00_ret_at_w(w,1e-5)
        ReG0f00[id] = ReG0f00_ret_at_w(w,1e-6)
    end 
    plot(w_points,ImG0f00,linestyle="dashed")
    plot(w_points,ReG0f00)
end 

function plot_Gf_at_00()
    Num_points = 1000
    w_points = LinRange(0.01,5.98,Num_points)
    Gf = zeros(Complex{Float64},Num_points)
    V00 = zeros(Num_points)
    for (id,w) in enumerate(w_points)
        if abs(abs(w)-2)<0.01
            w=1.98
        end 
        Gf[id] = (ReG0f00_ret_at_w(w,1e-5)+im*sign(w)ImG0f00_ret_at_w(w,1e-5))/(1+4*(ReG0f00_ret_at_w(w,1e-5)+im*sign(w)ImG0f00_ret_at_w(w,1e-5)))
        V00[id] = 1/abs(1+4*((ReG0f00_ret_at_w(w,1e-5)+im*sign(w)ImG0f00_ret_at_w(w,1e-5))))^2
        display(w)
    end 
    #plot(w_points,real(Gf))
    plot(w_points,imag(Gf),linestyle="dashed")
    plot(w_points,V00,color="b")
end 


function calculate_on_vison_pair_site_f_fermion_density(BCs)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    k_lattice = [(j1/L1)*g1 + (j2/L2 -(m*j1)/(L1*L2))*g2 for j1 = 0:(L1-1) for j2 = 0:(L2-1)]
    
    onsite_density = 0 

    for k in k_lattice
        abs_Delta_k , cos_phi_k = Delta_k(k)
        onsite_density += 0.5*(1+cos_phi_k)/(abs(1+4*(ReG0f00_ret_at_w(2*0.99*abs_Delta_k,1e-5)+im*ImG0f00_ret_at_w(2*0.99*abs_Delta_k,1e-5)))^2)
    end 

    return onsite_density/(L1*L2)
end

function calculate_off_site_vison_pair_site_f_density_matrix(BCs,site_n1n2)
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    k_lattice = [(j1/L1)*g1 + (j2/L2 -(m*j1)/(L1*L2))*g2 for j1 = 0:(L1-1) for j2 = 0:(L2-1)]
    r = site_n1n2[1]*a1+site_n1n2[2]*a2
    display(r)
    density = 0 

    #Im_w_points, ImG0f00_ret = get_ImG0f00_ret()
    #Re_w_points, ReG0f00_ret = get_ReG0f00_ret()

    for k in k_lattice
        abs_Delta_k , cos_phi_k = Delta_k(k)
        density += 0.5*(1+cos_phi_k)*(cos(dot(k,r)))/(abs(1+4*(Interpolate_func(2*0.99*abs_Delta_k,ReG0f00_ret,Re_w_points)+im*Interpolate_func(2*0.99*abs_Delta_k,ImG0f00_ret,Im_w_points)))^2)
    end 

    return density/(L1*L2)
end

function plot_Greens_function_for_f_fermions_off_site(BCs,r_n1_n2=[0,0])
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    k_lattice = [(j1/L1)*g1 + (j2/L2 -(m*j1)/(L1*L2))*g2 for j1 = 0:(L1-1) for j2 = 0:(L2-1)]
    r = r_n1_n2[1]*a1+r_n1_n2[2]*a2
    display(r)
    density = 0 

    E = zeros(L1*L2)
    Im_G_contribution = zeros(L1*L2)

    for (id,k) in enumerate(k_lattice)
        abs_Delta_k , cos_phi_k = Delta_k(k)

        E[id] = 2*abs_Delta_k
        Im_G_contribution[id] = pi*0.5*(1+cos_phi_k)*(cos(dot(k,r)))#/(abs(1+4*(Interpolate_func(2*0.99*abs_Delta_k,ReG0f00_ret,Re_w_points)+im*Interpolate_func(2*0.99*abs_Delta_k,ImG0f00_ret,Im_w_points)))^2)
        density += Im_G_contribution[id]
    end 

    Num_bins = 100
  
    w = reverse(LinRange(0,6,Num_bins))

    Im_G_w = zeros(Num_bins)

    binsize = 6/(Num_bins-1)

    for (bin_index,E_bin) in enumerate(w)
        display(E_bin)
        for j = 1:L1*L2 
            if E[j] > E_bin - binsize/2 && E[j] < E_bin + binsize/2 
                Im_G_w[bin_index] += Im_G_contribution[j]/(binsize)
            end
        end 
    end 

    Im_G_w[1] = 2*Im_G_w[1]
    Im_G_w = Im_G_w/(L1*L2)
    plot(w,Im_G_w)
    display(density/(L1*L2*pi))
    #plot(w,Re_G_w)
end



function ImG0f0r_ret_at_w(w,n1,n2,tol=1e-7)
    G0f0r_Integrand(theta) = (abs(w)/(8pi))*cos((n1-n2)*theta)*(1+sign(w)*((w^2+4-16*cos(theta)^2)/(4*abs(w))))*(((w^2+4)/8)*cos(theta)^2 - ((w^2-4)/16)^2 -cos(theta)^4)^(-1/2)
    if 2<abs(w)<6
        ImG0f0r,est = quadgk(G0f0r_Integrand,0,acos((abs(w)-2)/4),rtol=tol)
    elseif 0<abs(w)<2
        ImG0f0r,est = quadgk(G0f0r_Integrand,acos((2+abs(w))/4),acos((2-abs(w))/4),rtol=tol)
    end
    return ImG0f0r
end 

function plot_ImG0f0r_ret(n1,tol=1e-5)
    Num_points = 1000
    w_points = LinRange(-5.98,5.98,Num_points)
    ImG0f0r = zeros(Complex{Float64},Num_points)
    for (id,w) in enumerate(w_points)
        ImG0f0r[id] = ImG0f0r_ret_at_w(w,n1,-n1,tol)
    end 
    plot(w_points,ImG0f0r)
end 

function get_ImG0f0r_ret(n1,n2,Num_points=1000)
    ImG0f0r= zeros(2*Num_points)
    for (id,w) = enumerate(LinRange(0.01,5.9999,Num_points))
        ImG0f0r[Num_points+id] = ImG0f0r_ret_at_w(w,n1,n2,1e-5)
        ImG0f0r[Num_points+1-id] = ImG0f0r_ret_at_w(-w,n1,n2,1e-5)
        display(id)
    end
    w_points = [-reverse(collect(LinRange(0.01,5.99,Num_points)));collect(LinRange(0.01,5.99,Num_points))]
    return w_points, ImG0f0r
end 

function Interpolate_func(point,func,w_domain)
    if abs(point) > 5.97
        Interpolated_func = 0
    else
        b=findfirst((x -> x>point),w_domain)
        a=b-1
        Interpolated_func = func[a]+(point-w_domain[a])*(func[b]-func[a])/(w_domain[b]-w_domain[a])
    end
    return Interpolated_func
end

function Kramers_Kronig_at_w(w_point,func,w_domain,tol=1e-5)
    eta = 1e-6
    Integrand(w_prime) = (-1/pi)*Interpolate_func(w_prime,func,w_domain)*((w_prime-w_point)/((w_prime-w_point)^2+eta^2))
    KK_func,est = quadgk(Integrand,-5.97,5.97,rtol=tol)
    return KK_func
end

function plot_G0f0r_ret(n1,tol=1e-5)
    Num_points = 1000
    w_points = LinRange(-5.98,5.98,Num_points)
    ImG0f0r = zeros(Complex{Float64},Num_points)
    ImG0f00 = zeros(Complex{Float64},Num_points)
    ReG0f0r = zeros(Complex{Float64},Num_points)
    ReG0f00 = zeros(Complex{Float64},Num_points)
    ImGf0r = zeros(Complex{Float64},Num_points)
    for (id,w) in enumerate(w_points)
        ImG0f0r[id] = ImG0f0r_ret_at_w(w,n1,-n1,tol)
        ImG0f00[id] = ImG0f0r_ret_at_w(w,0,0,tol)
    end 
    for (id,w0) in enumerate(w_points)
        ReG0f0r[id] = Kramers_Kronig_at_w(w0,ImG0f0r,w_points)
        ReG0f00[id] = Kramers_Kronig_at_w(w0,ImG0f00,w_points)
        ImGf0r[id] = (ImG0f0r[id]+4*(ReG0f00[id]*ImG0f0r[id]-ReG0f0r[id]*ImG0f00[id]))/(abs(1+4*(ReG0f00[id]+im*ImG0f00[id]))^2)
    end 
    #plot(w_points,ReG0f0r)
    #plot(w_points,ImG0f0r,linestyle="dashed")
    plot(w_points,ImGf0r,linestyle="dashed")
end 

function get_Gf00_complex(w)
    BCs = [600,600,0]
    N = BCs[1]*BCs[2]
    L1 = BCs[1]
    L2 = BCs[2]
    m = BCs[3]

    ϵ = 0.015

    hl_lattice = [[j1/L1,(j2/L2 + (m*j1)/(L2*L1))] for j1 = 0:(L1-1) for j2 = 0:(L2-1)]

    k_lattice = [h[1]*g1+h[2]*g2 for h in hl_lattice]

    G_gtr = 0 
    G_lsr = 0 

    for k in k_lattice 
        Δk = 1+exp(im*dot(k,a1))+exp(im*dot(k,a2))
        G_gtr += (0.5*(1+cos(angle(Δk)))/(w-2*abs(Δk)+im*ϵ))/N
        G_lsr += (0.5*(1-cos(angle(Δk)))/(w+2*abs(Δk)-im*ϵ))/N
    end 

    G = -(G_gtr+G_lsr)
    return G
end 

function plot_G00_real_w(N=100)
    w_list = LinRange(-8,8,N)
    G = zeros(Complex{Float64},N)

    cmap = get_cmap("hsv")

    for (id,w) in enumerate(w_list) 
        G[id] = get_Gf00_complex(w)
        scatter(w,abs(G[id]),color=cmap((angle(G[id])+pi)/(2*pi)))
        sleep(0.05)
    end 
    Phase_colors = (angle.(G).+pi)/(2*pi)
    
    plot(w_list,imag.(G))
    plot(w_list,real.(G))
end 

function plot_G00_complex_w(N)
    Im_w_list = LinRange(-3,3,N)
    Re_w_list = LinRange(-2,2,N)

    cmap = get_cmap("hsv")

    for Re_w in Re_w_list
        for Im_w in Im_w_list
            G = get_Gf00_complex(Re_w+im*Im_w)
            scatter3D(Re_w,Im_w,abs(G),color=cmap((pi+angle(G))/(2*pi)))
            sleep(0.05)
        end 
    end 
end 

function plot_V_complex_w(N)
    Im_w_list = LinRange(-1,1,N)
    Re_w_list = LinRange(-7,-5,N)

    cmap = get_cmap("hsv")

    for Re_w in Re_w_list
        for Im_w in Im_w_list
            G = get_Gf00_complex(Re_w+im*Im_w)
            V = 1/(1+4*G)
            scatter3D(Re_w,Im_w,abs(V),color=cmap((pi+angle(V))/(2*pi)))
            sleep(0.05)
        end 
    end 
end 

function plot_V_real_w(N=100)
    w_list = LinRange(-8,8,N)
    V = zeros(Complex{Float64},N)

    cmap = get_cmap("hsv")

    for (id,w) in enumerate(w_list) 
        G = get_Gf00_complex(w)
        V[id] = 1/(1+4*G)
        display(V[id])
        scatter(w,abs(V[id]),color = cmap((angle(V[id])+pi)/(2*pi)))
        sleep(0.05)
    end 
    
    plot(w_list,imag.(V))
    plot(w_list,real.(V))
    #scatter(w_list,abs.(G),color=cmap(Phase_colors))
end 