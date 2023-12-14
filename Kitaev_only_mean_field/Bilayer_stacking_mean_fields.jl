#=
This is a specialised form of the code to calculate the mean fields for a Kitaev bilayer model focused on stacking arrangements
ASSUMPTIONS:
- No magnetic ordering
- Isotropic couplings (all bond types equivalent)

Date created 8/12/2023
=#
using HDF5
using LinearAlgebra
using PyPlot # This is the library used for plotting 
pygui(true) # This changes the plot backend from julia default to allow plots to be displayed

# sets the real space lattice vectors
a1 = [1/2, sqrt(3)/2]
a2 = [1/2, -sqrt(3)/2]

# sets the nearest neighbour vectors 
nx = (a1 - a2)/3
ny = (a1 + 2a2)/3
nz = -(2a1 + a2)/3

nn = [nx,ny,nz] # stores the nearest neighbours in a vector 

# sets the coupling parameters
K = -4

# defines "spin matrices" used throughout the calculation
Mx = [0 1 0 0 ;-1 0 0 0 ;0 0 0 -1; 0 0 1 0]
My = [0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0]
Mz = [0 0 0 1; 0 0 -1 0; 0 1 0 0 ; -1 0 0 0]
M_alpha = [Mx,My,Mz]

# defines "pseudospin matrices"
Gx = [0 -1 0 0 ; 1 0 0 0 ; 0 0 0 -1 ; 0 0 1 0]
Gy = [0 0 -1 0 ; 0 0 0 1 ; 1 0 0 0 ; 0 -1 0 0]
Gz = [0 0 0 -1 ; 0 0 -1 0 ; 0 1 0 0 ; 1 0 0 0]
G_alpha = [Gx,Gy,Gz]

# Sets the initial guesses
Ux_layer_1 = [0.5 0 0 0 ; 0 -0.5 0 0 ; 0 0 0 0 ; 0 0 0 0 ]
Ux_layer_2 = [-0.5 0 0 0 ; 0 0.5 0 0 ; 0 0 0 0 ; 0 0 0 0 ]
U_interlayer = [0.5 0 0 0 ; 0 0.2 0 0 ; 0 0 0.2 0 ; 0 0 0 0.2 ]
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
N = 16
g1, g2 = dual(a1,a2)
BZ = brillouinzone(g1,g2,N,false) # N must be even
half_BZ = brillouinzone(g1,g2,N,true)
Large_BZ = brillouinzone(g1,g2,100,false)

function transform_bond_type(mean_fields,alpha)
    """
    Given a 4x4 matrix of mean fields <iX_iX_j^T> calculated with a bond ij of type x, this transforms the mean fields to the equivalent matrix if it had been calculated with bond type alpha = x, y, z
    """
    Qx = diagm([1,1,1,1])
    Qy = [1 0 0 0 ; 0 0 1 0 ; 0 1 0 0 ; 0 0 0 1]
    Qz = [1 0 0 0 ; 0 0 0 1 ; 0 0 1 0 ; 0 1 0 0]
    Q = [Qx,Qy,Qz]

    return Q[alpha]*mean_fields*Q[alpha]
end

function fermi_dirac(E,temp=0.0,mu=0)
    """
    Given a scalar energy E it returns the probability for a fermionic state of that energy to be occupied
    """
    return (exp((E-mu)/temp) + 1)^-1
end

function diagonalise(H,temp=0.0,mu=0)
    """
    Diagonalises the Hamiltonian H
    returns:
    - a unitary matrix with eigenvectors as columns U
    - The occupancy matrix O giving the probability for the state to be occupied 
    """
    U = eigvecs(H)
    E = eigvals(H)
    O =  Diagonal(fermi_dirac.(E,temp,mu))
    return U , O 
end

function Kitaev_block(Ux_mean_fields,k)
    """
    creates a 4x4 matrix mathcal{M}^K which is a block in the Hamiltonian corresponing to the Kitaev interaction
    """
    MK = zeros(Complex{Float64},4,4)
    for alpha = [1,2,3]
        MK += -im*(K/4)*exp(im*dot(k,nn[alpha]))*M_alpha[alpha]*transform_bond_type(Ux_mean_fields,alpha)*M_alpha[alpha]
    end 
    return MK
end

function Kitaev_block_kitaev_rep(Ux_mean_fields,k)
    MK = zeros(Complex{Float64},4,4)
    for alpha = [1,2,3]
        MK += -im*(K/4)*exp(im*dot(k,nn[alpha]))*(M_alpha[alpha]-G_alpha[alpha])*transform_bond_type(Ux_mean_fields,alpha)*(M_alpha[alpha]-G_alpha[alpha])
    end 
    return MK
end


# This section is specialised to the monolayer 
function update_monolayer_mean_fields(Ux_layer_1)
    new_Ux_1 = zeros(Complex{Float64},4,4)
    for k in half_BZ
        MK = Kitaev_block(Ux_layer_1,k)
        H_K = [zeros(4,4) MK ; MK' zeros(4,4)]
        R, O = diagonalise(H_K)
        all_fields = R*(O)*R'
        new_Ux_1 += 2*imag.(exp(-im*dot(k,nn[1]))*all_fields)[1:4,5:8]
    end 

    return Diagonal(diag(new_Ux_1))/N^2
end 

function interlayer_block(U_interlayer_mean_fields,J_perp)
    M_perp = zeros(Complex{Float64},4,4)
    for alpha = [1,2,3]
        M_perp += -im*(J_perp/4)*M_alpha[alpha]*U_interlayer_mean_fields*M_alpha[alpha]
    end 
    return M_perp
end 

function interlayer_block_kitaev_rep(U_interlayer_mean_fields,J_perp)
    M_perp = zeros(Complex{Float64},4,4)
    for alpha = [1,2,3]
        M_perp += -im*(J_perp/4)*(M_alpha[alpha]-G_alpha[alpha])*U_interlayer_mean_fields*(M_alpha[alpha]-G_alpha[alpha])
    end 
    return M_perp
end 

function AA_Hamiltonian(Ux_layer_1,Ux_layer_2,U_interlayer_A,U_interlayer_B,k,J_perp)
    MK1 = Kitaev_block(Ux_layer_1,k)
    MK2 = Kitaev_block(Ux_layer_2,k)
    M_perp_A = interlayer_block(U_interlayer_A,J_perp)
    M_perp_B = interlayer_block(U_interlayer_B,J_perp)

    z = zeros(4,4)
    H_AA = [z MK1 M_perp_A z ; MK1' z z M_perp_B ; M_perp_A' z z MK2 ; z M_perp_B' MK2' z]
    return H_AA
end 

function AA_Hamiltonian_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer_A,U_interlayer_B,k,J_perp)
    MK1 = Kitaev_block_kitaev_rep(Ux_layer_1,k)
    MK2 = Kitaev_block_kitaev_rep(Ux_layer_2,k)
    M_perp_A = interlayer_block_kitaev_rep(U_interlayer_A,J_perp)
    M_perp_B = interlayer_block_kitaev_rep(U_interlayer_B,J_perp)

    z = zeros(4,4)
    H_AA = [z MK1 M_perp_A z ; MK1' z z M_perp_B ; M_perp_A' z z MK2 ; z M_perp_B' MK2' z]
    return H_AA
end 

function AB_Hamiltonian(Ux_layer_1,Ux_layer_2,U_interlayer,k,J_perp=0.0)
    MK1 = Kitaev_block(Ux_layer_1,k)
    MK2 = Kitaev_block(Ux_layer_2,k)
    M_perp = interlayer_block(U_interlayer,J_perp)

    H_AB = [zeros(4,4) MK1 zeros(4,4) M_perp ; MK1' zeros(4,4) zeros(4,4) zeros(4,4) ; zeros(4,4) zeros(4,4) zeros(4,4) transpose(MK2) ; M_perp' zeros(4,4) conj(MK2) zeros(4,4)]
    return H_AB 
end 

function AB_Hamiltonian_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer,k,J_perp=0.0)
    MK1 = Kitaev_block_kitaev_rep(Ux_layer_1,k)
    MK2 = Kitaev_block_kitaev_rep(Ux_layer_2,k)
    M_perp = interlayer_block_kitaev_rep(U_interlayer,J_perp)

    H_AB = [zeros(4,4) MK1 zeros(4,4) M_perp ; MK1' zeros(4,4) zeros(4,4) zeros(4,4) ; zeros(4,4) zeros(4,4) zeros(4,4) transpose(MK2) ; M_perp' zeros(4,4) conj(MK2) zeros(4,4)]
    return H_AB 
end 

function update_AB_mean_fields(Ux_layer_1,Ux_layer_2,U_interlayer,J_perp)
    new_Ux_1 = zeros(Complex{Float64},4,4)
    new_Ux_2 = zeros(Complex{Float64},4,4)
    new_U_interlayer = zeros(Complex{Float64},4,4)
    for k in half_BZ
        H_AB = AB_Hamiltonian(Ux_layer_1,Ux_layer_2,U_interlayer,k,J_perp)
        R, O = diagonalise(H_AB)
        all_fields = conj(R)*O*conj(R')
        new_Ux_1 += exp(-im*dot(k,nn[1]))*all_fields[1:4,5:8]
        new_Ux_2 += exp(-im*dot(k,nn[1]))*all_fields[9:12,13:16]
        new_U_interlayer += all_fields[1:4,13:16]
    end 

    return imag.(new_Ux_1/N^2), imag.(new_Ux_2/N^2) , imag.(new_U_interlayer/N^2)
end 

function update_AB_mean_fields_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer,J_perp)
    new_Ux_1 = zeros(4,4)
    new_Ux_2 = zeros(4,4)
    new_U_interlayer = zeros(4,4)
    for k in half_BZ
        H_AB = AB_Hamiltonian_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer,k,J_perp)
        R, O = diagonalise(H_AB)
        all_fields = R*O*R'
        new_Ux_1 += 2*imag.(exp(-im*dot(k,nn[1]))*all_fields[1:4,5:8])
        new_Ux_2 += 2*imag.(exp(-im*dot(k,nn[1]))*all_fields[9:12,13:16])
        new_U_interlayer += 2*imag.(all_fields[1:4,13:16])
    end 

    return -Diagonal(diag(new_Ux_1))/N^2, -Diagonal(diag(new_Ux_2))/N^2 , -Diagonal(diag(new_U_interlayer))/N^2
end 

function update_AA_mean_fields_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer_A,U_interlayer_B,J_perp)
    new_Ux_1 = zeros(4,4)
    new_Ux_2 = zeros(4,4)
    new_U_interlayer_A = zeros(4,4)
    new_U_interlayer_B = zeros(4,4)
    for k in half_BZ
        H_AA = AA_Hamiltonian_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer_A,U_interlayer_B,k,J_perp)
        R, O = diagonalise(H_AA)
        all_fields = R*O*R'
        new_Ux_1 += 2*imag.(exp(-im*dot(k,nn[1]))*all_fields)[1:4,5:8]
        new_Ux_2 += 2*imag.(exp(-im*dot(k,nn[1]))*all_fields)[9:12,13:16]
        new_U_interlayer_A += 2*imag(all_fields)[1:4,9:12]
        new_U_interlayer_B += 2*imag(all_fields)[5:8,13:16]
    end 

    return -Diagonal(diag(new_Ux_1))/N^2, Diagonal(diag(new_Ux_1))/N^2 , -Diagonal(diag(new_U_interlayer_A))/N^2 , -Diagonal(diag(new_U_interlayer_A))/N^2
end 

function run_to_convergence_AA_kitaev_rep(initial_Ux_1,initial_Ux_2,initial_U_perp_A,initial_U_perp_B,J_perp,tolerance=10.0,tol_drop_iteration=500)

    initial_mean_fields = [initial_Ux_1 initial_Ux_2 initial_U_perp_A initial_U_perp_B]
    old_mean_fields = initial_mean_fields
    new_Ux_1 = zeros(4,4)
    new_Ux_2 = zeros(4,4)
    new_U_perp_A = zeros(4,4)
    new_U_perp_B = zeros(4,4)
    new_mean_fields = [new_Ux_1 new_Ux_2 new_U_perp_A new_U_perp_B]

    #old_old_mean_fields = zeros(8,8)
    tol = 10^(-tolerance)
    it_num = 0 
    osc_num = 0
    not_converged = true
    not_oscillating = true 
    mark_with_x = false
    while not_converged
        new_Ux_1,new_Ux_2,new_U_perp_A, new_U_perp_B = update_AA_mean_fields_kitaev_rep(old_mean_fields[1:4,1:4],old_mean_fields[1:4,5:8],old_mean_fields[1:4,9:12],old_mean_fields[1:4,13:16],J_perp)
        new_mean_fields = [new_Ux_1 new_Ux_2 new_U_perp_A new_U_perp_B]
        diff= abs.(new_mean_fields-old_mean_fields)
        #display(diff)
        #display(new_mean_fields)
        #sleep(0.5)
        #diff2 = abs.(new_mean_fields - old_old_mean_fields)
        not_converged = any(diff .> tol)
        println(it_num)

        #=
        not_oscillating = any(diff2 .> 0.01*tol)
        if not_oscillating == false
            osc_num += 1
            println("osc number is $osc_num")
            display(diff.*(diff .>tol))
            display(diff2.*(diff2 .>tol))
        else
            osc_num = 0
        end

        if osc_num >= 10
            println("Oscillating solution")
            mark_with_x = true
            break
        end
        =#
        if it_num%tol_drop_iteration == 0 && it_num >0
            tol = 10*tol
            println(tol)
            mark_with_x = true
        end

        it_num +=1 
        #old_old_mean_fields = old_mean_fields
        old_mean_fields = new_mean_fields
    end
    return round.(new_mean_fields[1:4,1:4],digits=trunc(Int,tolerance)) ,round.(new_mean_fields[1:4,5:8],digits=trunc(Int,tolerance)),round.(new_mean_fields[1:4,9:12],digits=trunc(Int,tolerance)),round.(new_mean_fields[1:4,13:16],digits=trunc(Int,tolerance))
end

function run_to_convergence_AB(initial_Ux_1,initial_Ux_2,initial_U_perp,J_perp,tolerance=10.0,tol_drop_iteration=500)

    initial_mean_fields = [initial_Ux_1 initial_Ux_2 initial_U_perp]
    old_mean_fields = initial_mean_fields
    new_mean_fields = old_mean_fields
    new_Ux_1 = zeros(4,4)
    new_Ux_2 = zeros(4,4)
    new_U_perp = zeros(4,4)

    #old_old_mean_fields = zeros(8,8)
    tol = 10^(-tolerance)
    it_num = 0 
    osc_num = 0
    not_converged = true
    not_oscillating = true 
    mark_with_x = false
    while not_converged
        new_Ux_1,new_Ux_2,new_U_perp = update_mean_fields(old_mean_fields[1:4,1:4],old_mean_fields[1:4,5:8],old_mean_fields[1:4,9:12],J_perp)
        new_mean_fields = [new_Ux_1 new_Ux_2 new_U_perp]
        diff= abs.(new_mean_fields-old_mean_fields)
        display(new_mean_fields)
        sleep(0.5)
        #diff2 = abs.(new_mean_fields - old_old_mean_fields)
        not_converged = any(diff .> tol)
        println(it_num)

        #=
        not_oscillating = any(diff2 .> 0.01*tol)
        if not_oscillating == false
            osc_num += 1
            println("osc number is $osc_num")
            display(diff.*(diff .>tol))
            display(diff2.*(diff2 .>tol))
        else
            osc_num = 0
        end

        if osc_num >= 10
            println("Oscillating solution")
            mark_with_x = true
            break
        end
        =#
        if it_num%tol_drop_iteration == 0 && it_num >0
            tol = 10*tol
            println(tol)
            mark_with_x = true
        end

        it_num +=1 
        #old_old_mean_fields = old_mean_fields
        old_mean_fields = new_mean_fields
    end
    new_Ux_1 = new_mean_fields[1:4,1:4]
    new_Ux_2 = new_mean_fields[1:4,5:8]
    new_U_interlayer = new_mean_fields[1:4,9:12]
    return round.(new_Ux_1,digits=trunc(Int,tolerance)) ,round.(new_Ux_2,digits=trunc(Int,tolerance)),round.(new_U_interlayer,digits=trunc(Int,tolerance))
end

function run_to_convergence_AB_kitaev_rep(initial_Ux_1,initial_Ux_2,initial_U_perp,J_perp,tolerance=10.0,tol_drop_iteration=500)

    initial_mean_fields = [initial_Ux_1 initial_Ux_2 initial_U_perp]
    old_mean_fields = initial_mean_fields
    new_mean_fields = old_mean_fields
    new_Ux_1 = zeros(4,4)
    new_Ux_2 = zeros(4,4)
    new_U_perp = zeros(4,4)

    #old_old_mean_fields = zeros(8,8)
    tol = 10^(-tolerance)
    it_num = 0 
    osc_num = 0
    not_converged = true
    not_oscillating = true 
    mark_with_x = false
    while not_converged
        new_Ux_1,new_Ux_2,new_U_perp = update_AB_mean_fields_kitaev_rep(old_mean_fields[1:4,1:4],old_mean_fields[1:4,5:8],old_mean_fields[1:4,9:12],J_perp)
        new_mean_fields = [new_Ux_1 new_Ux_2 new_U_perp]
        diff= abs.(new_mean_fields-old_mean_fields)
        #display(new_mean_fields)
        #sleep(0.5)
        #diff2 = abs.(new_mean_fields - old_old_mean_fields)
        not_converged = any(diff .> tol)
        println(it_num)

        #=
        not_oscillating = any(diff2 .> 0.01*tol)
        if not_oscillating == false
            osc_num += 1
            println("osc number is $osc_num")
            display(diff.*(diff .>tol))
            display(diff2.*(diff2 .>tol))
        else
            osc_num = 0
        end

        if osc_num >= 10
            println("Oscillating solution")
            mark_with_x = true
            break
        end
        =#
        if it_num%tol_drop_iteration == 0 && it_num >0
            tol = 10*tol
            println(tol)
            mark_with_x = true
        end

        it_num +=1 
        #old_old_mean_fields = old_mean_fields
        old_mean_fields = new_mean_fields
    end
    new_Ux_1 = new_mean_fields[1:4,1:4]
    new_Ux_2 = new_mean_fields[1:4,5:8]
    new_U_interlayer = new_mean_fields[1:4,9:12]
    return round.(new_Ux_1,digits=trunc(Int,tolerance)) ,round.(new_Ux_2,digits=trunc(Int,tolerance)),round.(new_U_interlayer,digits=trunc(Int,tolerance))
end

function scan_J_perp_for_AB_stacking(J_perp_min,J_perp_max,num_points)
    J_perp_points = LinRange(J_perp_min,J_perp_max,num_points)
    for J_perp in J_perp_points
        Ux_1 ,Ux_2, U_perp = run_to_convergence_AB(Ux_layer_1,Ux_layer_2,U_interlayer,J_perp,3.0)
        U00_1 = Ux_1[1,1]
        U11_1 = Ux_1[2,2]
        U00_2 = Ux_2[1,1]
        U11_2 = Ux_2[2,2]
        U00_perp = U_perp[1,1]
        U11_perp = U_perp[2,2]
        scatter(J_perp,U00_1,color="b")
        scatter(J_perp,U11_1,color="g")
        scatter(J_perp,U00_perp,color="r")
        scatter(J_perp,U11_perp,color="purple")
    end 
end 


function scan_J_perp_for_AA_stacking_kitaev_rep(J_perp_min,J_perp_max,num_points)
    """
    J_perp is measured in units of the Kitaev coupling set to K=4 by default 
    """
    J_perp_points = -K*LinRange(J_perp_min,J_perp_max,num_points)
    for J_perp in J_perp_points
        display(J_perp)
        Ux1 ,Ux2, UpA, UpB = run_to_convergence_AA_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer, U_interlayer,J_perp,4.0)
        U00_1 = Ux1[1,1]
        U11_1 = Ux1[2,2]
        U00_2 = Ux2[1,1]
        U11_2 = Ux2[2,2]
        U00_pA = UpA[1,1]
        U11_pA = UpA[2,2]
        scatter(J_perp/abs(K),U00_1,color="b")
        scatter(J_perp/abs(K),U00_pA,color="r")
        scatter(J_perp/abs(K),U11_pA,color="g")
        scatter(J_perp/abs(K),U11_1,color="purple")
    end 
end

function scan_J_perp_for_AB_stacking_kitaev_rep(J_perp_min,J_perp_max,num_points)
    """
    J_perp is measured in units of the Kitaev coupling set to K=4 by default 
    """
    J_perp_points = -K*LinRange(J_perp_min,J_perp_max,num_points)
    for J_perp in J_perp_points
        display(J_perp)
        Ux1 ,Ux2, Up = run_to_convergence_AB_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer,J_perp,4.0)
        U00_1 = Ux1[1,1]
        U11_1 = Ux1[2,2]
        U00_2 = Ux2[1,1]
        U11_2 = Ux2[2,2]
        U00_p = Up[1,1]
        U11_p = Up[2,2]
        scatter(J_perp/abs(K),U00_1,color="b")
        scatter(J_perp/abs(K),U00_p,color="r")
        scatter(J_perp/abs(K),U11_p,color="g")
        scatter(J_perp/abs(K),U11_1,color="purple")
    end 
end

function scan_J_perp_for_AB_stacking_kitaev_rep(J_perp_min,J_perp_max,num_points)
    """
    J_perp is measured in units of the Kitaev coupling set to K=4 by default 
    """
    J_perp_points = -K*LinRange(J_perp_min,J_perp_max,num_points)
    for J_perp in J_perp_points
        display(J_perp)
        Ux1 ,Ux2, Up = run_to_convergence_AB_kitaev_rep(Ux_layer_1,Ux_layer_2,U_interlayer,J_perp,4.0)
        U00_1 = Ux1[1,1]
        U11_1 = Ux1[2,2]
        U00_2 = Ux2[1,1]
        U11_2 = Ux2[2,2]
        U00_p = Up[1,1]
        U11_p = Up[2,2]
        scatter(J_perp/abs(K),U00_1,color="b")
        scatter(J_perp/abs(K),U00_p,color="r")
        scatter(J_perp/abs(K),U11_p,color="g")
        scatter(J_perp/abs(K),U11_1,color="purple")
    end 
end

function scan_J_perp_for_AB_stacking_kitaev_rep(J_perp_min,J_perp_max,num_points)
    """
    J_perp is measured in units of the Kitaev coupling set to K=4 by default 
    """
    J_perp_points = -K*LinRange(J_perp_min,J_perp_max,num_points)
    for J_perp in J_perp_points
        display(J_perp)
        Ux1 ,Ux2, Up = run_to_convergence_AB(Ux_layer_1,Ux_layer_2,U_interlayer,J_perp,4.0)
        U00_1 = Ux1[1,1]
        U11_1 = Ux1[2,2]
        U00_2 = Ux2[1,1]
        U11_2 = Ux2[2,2]
        U00_p = Up[1,1]
        U11_p = Up[2,2]
        scatter(J_perp/abs(K),U00_1,color="b")
        scatter(J_perp/abs(K),U00_p,color="r")
        scatter(J_perp/abs(K),U11_p,color="g")
        scatter(J_perp/abs(K),U11_1,color="purple")
    end 
end