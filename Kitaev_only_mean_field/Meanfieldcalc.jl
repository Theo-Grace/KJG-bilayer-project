using LinearAlgebra
using PyPlot # This is the library used for plotting 
pygui(true) # This changes the plot backend from julia default to allow plots to be displayed

#This sets the real space lattice vectors
a1 = [1/2, sqrt(3)/2]
a2 = [1/2, -sqrt(3)/2]

nx = (a1 - a2)/3
ny = (a1 + 2a2)/3
nz = -(2a1 + a2)/3

nn = [nx,ny,nz] # stores the nearest neighbours in a vector 

# sets the Kitaev coupling parameters
K = [1,1,1]

function antisymmetrise(A)
    """
    Takes a square matrix A and returns a matrix with the lower triangle set to minus the upper triangle of A 
    """
    dim = convert(Int,sqrt(length(A)))
    for i in 2:dim
        for j in 1:i-1
            A[i,j]=-A[j,i]
        end
    end
    return A 
end

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
N = 50
g1, g2 = dual(a1,a2)
BZ = brillouinzone(g1,g2,N,false) # N must be even 

function initial_guess_mean_fields(uC=-1,uK=1,uJ=0)
    mean_fields = zeros(8,8,3)

    mean_fields[1,5,1] = uC
    mean_fields[1,5,2] = uC 
    mean_fields[1,5,3] = uC 
    mean_fields[2,6,1] = uK
    mean_fields[3,7,2] = uK
    mean_fields[4,8,3] = uK
    mean_fields[2,6,2] = uJ
    mean_fields[2,6,3] = uJ
    mean_fields[3,7,1] = uJ
    mean_fields[3,7,3] = uJ
    mean_fields[4,8,1] = uJ
    mean_fields[4,8,2] = uJ
    for n = 1:3
        mean_fields[:,:,n] = antisymmetrise(mean_fields[:,:,n])
    end

    return mean_fields
end

function Hamiltonian(mean_fields,k,nn,K=[1,1,1])
    """
    Calculates the Hamiltonian for a given wavevector as an 8x8 matrix
    This assumes only Kitaev coupling, no Heisenberg or Gamma terms 
    Requires:
    - an 8x8x3 matrix of mean fields in real space <iX_aX_a^T>. 
    The 3rd dimension specifies the direction of nearest neighbours alpha = x,y,z
    """

    M = zeros(Complex{Float64},8,8,3,3) # 3rd dimension specifies which majoranas are coupled, 4th dimension specifies bond type 
    H = zeros(Complex{Float64},8,8)
    
    for alpha = 1:3
        for beta = 1:3
            M[1,1+alpha,alpha,beta] = mean_fields[5,5+alpha,beta] # adds term associate to mean spin on site j <S_j>
            M[5,5+alpha,alpha,beta] = mean_fields[1,1+alpha,beta] # adds term associated to mean spin on site i <S_i>
            M[1,5,alpha,beta] = -mean_fields[1+alpha,5+alpha,beta] # adds term associated to gauge sector majoranas u^alpha_ij 
            M[1+alpha,5+alpha,alpha,beta] = -mean_fields[1,5,beta] # adds term associated to matter sector majoranas u^0_ij
            M[1,5+alpha,alpha,beta] = mean_fields[1+alpha,5,beta] # adds sector mixing term m'_ij
            M[1+alpha,5,alpha,beta] = mean_fields[1,5+alpha,beta] # adds sector mixing term m _ij
            M[:,:,alpha,beta] = antisymmetrise(M[:,:,alpha,beta])
        end
        H = H -im*0.5*K[alpha]*Fourier(M[:,:,alpha,alpha],k,nn[alpha])
    end
    return H 
end

function Hamiltonian_J(mean_fields,k,nn,J=1)
    """
    Given a set of mean fields as an 8x8x3 matrix calculates the Heisenberg term for the Hamiltonian 
    as a function of wavevector k 
    """
    M = zeros(Complex{Float64},8,8,3,3) # 3rd dimension specifies which majoranas are coupled, 4th dimension specifies bond type 
    H_J = zeros(Complex{Float64},8,8)
    
    for alpha = 1:3
        for beta = 1:3
            M[1,1+alpha,alpha,beta] = mean_fields[5,5+alpha,beta]
            M[5,5+alpha,alpha,beta] = mean_fields[1,1+alpha,beta]
            M[1,5,alpha,beta] = -mean_fields[1+alpha,5+alpha,beta]
            M[1+alpha,5+alpha,alpha,beta] = -mean_fields[1,5,beta]
            M[1,5+alpha,alpha,beta] = mean_fields[1+alpha,5,beta]
            M[1+alpha,5,alpha,beta] = mean_fields[1,5+alpha,beta]
            M[:,:,alpha,beta] = antisymmetrise(M[:,:,alpha,beta])

            H_J += -0.5im*J*Fourier(M[:,:,alpha,beta],k,nn[beta])
        end
    end
    return H_J
end 

function Hamiltonian_G(mean_fields,k,nn,G=1)
    """
    """
    M_G = zeros(Complex{Float64},8,8,3)
    H_G = zeros(Complex{Float64},8,8)

    for alpha = 1:3
        for beta = setdiff([1,2,3],alpha)
            M_G[1,1+beta,alpha] = mean_fields[,,alpha]
        end
    end
end

function Fourier(M,k,neighbour_vector)
    """
    Given an 8x8 matrix M this returns the matrix with off diagonal 4x4 blocks multiplied by phase factors e^ik.n
    This implements a Fourier transform from mean fields in real space to momentum space
    """
    phase = exp(im*dot(k,neighbour_vector))
    F = Diagonal([1,1,1,1,phase,phase,phase,phase])
    return (F')*M*F
end

function fermi_dirac(E,temp=0.0,mu=0)
    """
    Given a scalar energy E it returns the probability for a fermionic state of that energy to be occupied
    """
    return (exp((E-mu)/temp) + 1)^-1
end

function diagonalise(H)
    """
    Diagonalises the Hamiltonian H
    returns:
    - a unitary matrix with eigenvectors as columns U
    - The occupancy matrix O giving the probability for the state to be occupied 
    """
    U = eigvecs(H)
    E = eigvals(H)
    O =  Diagonal(fermi_dirac.(E))
    return U , O 
end

function filter(M,tolerance=14.0)
    """
    Takes:
    - a complex matrix M of any size
    - a tolerance such that any element smaller than 10^(-tolerance) will be set to 0
    Splits the matrix into real and imaginary components
    returns:
    - a filtered matrix of the same size as M with all terms less than the tolerance removed 
    """
    Magnitude = abs.(M)
    real_M = real.(M)
    imag_M = imag.(M)
    real_filtered_M = (.~((abs.(real_M).>0).&&(abs.(real_M).<10^(-tolerance)))).*real_M
    imag_filtered_M = (.~((abs.(imag_M).>0).&&(abs.(imag_M).<10^(-tolerance)))).*imag_M
    return real_filtered_M + im*imag_filtered_M
end

function update_mean_fields(BZ,old_mean_fields,nn,K)
    """
    calculates an updated mean field matrix. This calculates the Hamiltonian from a given set of mean fields, then diagonalises the Hamiltonian and calculates a set of mean fields from that Hamiltonian
    Requires:
    - a current mean field 8x8x3 matrix 
    returns
    - a new mean field 8x8x3 matrix
    """
    Num_unit_cells = length(BZ)
    updated_mean_fields = zeros(Complex{Float64},8,8,3)
    for alpha in 1:3
        for k in BZ
            H = Hamiltonian(old_mean_fields,k,nn,K)
            U , occupancymatrix = diagonalise(H)
            updated_mean_fields[:,:,alpha] += Fourier(transpose(U')*occupancymatrix*transpose(U),k,nn[alpha])
        end
    end
    updated_mean_fields = (im.*updated_mean_fields)./Num_unit_cells

    return updated_mean_fields
end

function run_to_convergence(BZ,initial_mean_fields,nn,tolerance=10.0,K=[1,1,1])
    old_mean_fields = initial_mean_fields
    new_mean_fields = update_mean_fields(BZ,old_mean_fields,nn,K)
    difference = new_mean_fields-old_mean_fields
    it_num = 0 
    while filter(difference,tolerance) != zeros(Complex{Float64},8,8,3)
        old_mean_fields = new_mean_fields
        new_mean_fields = update_mean_fields(BZ,old_mean_fields,nn,K)
        difference= new_mean_fields-old_mean_fields
        it_num +=1 
        println(it_num)
    end
    return round.(filter(new_mean_fields,tolerance),digits=trunc(Int,tolerance))
end

function run_for_fixed_number_of_steps(BZ,initial_mean_fields,nn,steps=100,K=[1,1,1])
    old_mean_fields = initial_mean_fields
    new_mean_fields = update_mean_fields(BZ,old_mean_fields,nn,K)
    difference = new_mean_fields-old_mean_fields
    println(1)
    for step_num in 2:steps
        old_mean_fields = new_mean_fields
        new_mean_fields = update_mean_fields(BZ,old_mean_fields,nn,K)
        difference = new_mean_fields-old_mean_fields
        if filter(difference,10.0) == zeros(Complex{Float64},8,8,3)
            println("Converged")
            return new_mean_fields
            break
        end
        println(step_num)
    end 
    display(difference)
    return new_mean_fields
end



# This section is used to calculate the analytical result for the mean fields to compare to numerical calculations. 
function phi(k,nn)
    p = 0
    for alpha = 1:3
        p += K[alpha]*exp(im*dot(k,nn[alpha]))
    end
    return angle(p)
end

function get_uC_exactly(BZ,alpha=1)
    """
    Calculates the analytical result for the mean field u0 given by eq 14a) of "Bilayer Kitaev models - Siefert"  
    """
    u0_exact=0
    for k in BZ
        u0_exact += cos(phi(k,nn)-dot(k,nn[alpha]))
    end
    return u0_exact/(2*length(BZ))
end

# This section calculates the bandstructure given a set of mean fields by calculating the eigenvalues of the corresponding Hamiltonian for each wavevector.
# These functions can be used to plot interesting directions in the brillouin zone 
function get_bandstructure(BZ,mean_fields,nn,K=[1,1,1])
    """
    takes:
    - The BZ as a matrix of k vectors
    - an 8x8x3 matrix of mean fields mean_fields 
    - nearest neighbour vectors in a vector nn
    - Kitaev coupling parameters K
    returns:
    - A dictionary called bandstructure with a key for each k in the Brillouin zone whose entries are a vector
    containing the energies for that k 
    """
    bandstructure = Dict()
    for k in BZ 
        H = Hamiltonian(mean_fields,k,nn,K)
        bandstructure[k] = eigvals(H)
    end
    return bandstructure
end

function plot_bands_G_to_K(BZ,bandstructure)
    """
    This plots the bands along the direction between Gamma and M points in the Brillouin Zone
    This requires:
    - the bandstructure as a dictionary bandstructure[k] whose entries are the 8 energies for that k vector
    - the Brillouin zone BZ as a matrix of k vectors 
    """
    GtoK = []
    bands_GtoK = [[] for i = 1:8]
    for k in BZ 
        if (k[2] == 0) 
            push!(GtoK,k)
            for i in 1:8
                push!(bands_GtoK[i],bandstructure[k][i])
            end
        end 
    end 
    kGtoK = (1:length(GtoK))*(g1[1]+g2[1])/(2*length(GtoK))
    for i in 1:8
        plot(kGtoK,bands_GtoK[i])
    end
    title("Majorana bandstructure between \$\\Gamma\$ and \$ K \$ points")
    ylabel("Energy")
    xlabel("Wavevector")
end

function plot_bands_G_to_M(BZ,bandstructure)
    """
    This plots the bands along the direction between Gamma and K points in the BZ 
    This requires:
    - the bandstructure as a dictionary bandstructure[k] whose entries are the 8 energies at that k value
    - the Brillouin zone BZ as a matrix of k vectors 
    """
    GtoM =[]
    bands_GtoM = [[] for i = 1:8]
    for k in BZ 
        if (k[1]==0)
            push!(GtoM,k)
            for i in 1:8
                push!(bands_GtoM[i],bandstructure[k][i])
            end
        end 
    end 
    for i in 1:8
        plot(1:length(GtoM),bands_GtoM[i])
    end
    title("Majorana bandstructure between \$\\Gamma\$ and \$ M \$ points")
    ylabel("Energy")
    xlabel("Wavevector")
    #legend(["matter","gauge","gauge","gauge","gauge","gauge","gauge","matter"])
end

function plot_BZ(BZ)
    """
    This will plot the vectors in k space given a BZ as a matrix of k vectors
    """
    N = sqrt(length(BZ))
    N = convert(Int,N)
    kx = zeros(N^2)
    ky = zeros(N^2)
    for (idx,k) in enumerate(BZ)
        kx[idx] = k[1]
        ky[idx] = k[2]
    end
    scatter(kx,ky)
    scatter(g1[1],g1[2])
end