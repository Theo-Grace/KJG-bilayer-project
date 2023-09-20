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
    b=findfirst((x -> x>w),w_points)
    a=b-1
    Interpolated_ImG0AA = ImG0AA[a]+(w-w_points[a])*(ImG0AA[b]-ImG0AA[a])/(w_points[b]-w_points[a])
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

function save_ReG0AA_data(w_points,ReG0AA,tol)
    Num_points = size(w_points)[1]/2

    Description = "The number of points used to calculate ReG0AA was $Num_points 
    The tolerance for numerical integration was tol=$tol.
    The Integration was perfomed using quadgk"

    group_name = "ReG0AA_data"

    fid = h5open(homedir()*"\\Documents\\Physics\\Kitaev model project\\G0AA","cw")
    it_num = 2
    while group_name in keys(fid)
        global group_name = group_name*"_$it_num"
    end

    create_group(fid,group_name)
    g = fid[group_name]
    write(g,"Description",Description)
    write(g,"w_points",w_points)
    write(g,"Real part of G0AA",ReG0AA)
    close(fid)
end
# Note that ReG0AA ~< 0.1 for |w|>15 

function load_ReG0AA()
    fid = h5open(homedir()*"\\OneDrive - The University of Manchester\\Physics work\\PhD\\Majorana Green function data\\G0AA\\ReG0AA_data","r")
    group_name = "ReG0AA_data_2"
    g = fid[group_name]
    show(stdout,"text/plain",keys(fid))
    show(stdout,"text/plain",keys(g))
    w_points = read(g["w_points"])
    ReG0AA = read(g["Real part of G0AA"])
    close(fid)

    return w_points, ReG0AA
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

function get_ReG0AB(Num_points=10)
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
    b=findfirst((x -> x>w),ReG0AB_w_points)
    a=b-1
    Interpolated_ReG0AB = ReG0AB[a]+(w-ReG0AB_w_points[a])*(ReG0AB[b]-ReG0AB[a])/(ReG0AB_w_points[b]-ReG0AB_w_points[a])
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
