# These are numerical calculations used to calculate the Matter sector ground state for the case of open flux pairs 
# The functions used come from Bilayer_Variationalcalc_2.jl 

# Sets the boundary conditions 
BCs = [25,25,0]


# gets the Matter sector Hamiltonian for fluxless and open pair flux states 
M0 = get_M0(BCs)

Mv = flip_bond_variable(M0,BCs,[1,1],"y")
Mv = flip_bond_variable(Mv,BCs,[2,1],"x")

# gets the matter sector Hamiltonian for a "closed pair" 
Mvp = flip_bond_variable(M0,BCs,[1,1],"z")

Uvp ,Vvp = get_U_and_V(Mvp)

F_closed = (U0'*Uvp)*(V0'*Vvp)'

# SIngular value decomposition and polar matrices
U0 ,V0 = get_U_and_V(M0)
Uv ,Vv = get_U_and_V(Mv)
Fv = Uv*Vv'
F0 = U0*V0'

F = (U0'*Uv)*(V0'*Vv)'

Q = eigvecs(F)
z = Q[:,1]

s0 = convert_n1_n2_to_site_index([1,1],BCs)
s1 = convert_n1_n2_to_site_index([2,1],BCs)
s2 = convert_n1_n2_to_site_index([1,0],BCs)

O = angle(eigvals(F)[1])

Λ2 = 2(cos(O)-1)*(transpose(z)*U0'*M0*V0*z-2*((U0*z)[s1]*(V0*z)[s0]+(U0*z)[s0]*(V0*z)[s2])) 

ϕ = angle(Λ2)


z = exp(-im*ϕ/2)*z
a = sqrt(2)*real.(z)
b = sqrt(2)*imag.(z)

# Useful quantities
Ua1 = (U0*a)[s1]
Ub1 = (U0*b)[s1] 
Va1 = (V0*a)[s1]
Vb1 = (V0*b)[s1]

Ua0 = (U0*a)[s0]
Ub0 = (U0*b)[s0] 
Va0 = (V0*a)[s0]
Vb0 = (V0*b)[s0]

Ua2 = (U0*a)[s2]
Ub2 = (U0*b)[s2] 
Va2 = (V0*a)[s2]
Vb2 = (V0*b)[s2]


# Potential term on x bond
v1_exact = 2*(Fv[s1,s0]-F0[s1,s0])  # Exact <0^2|V1|0^2> - <0^1|V1|0^1> 
V1_est = -2*sin(O)*(Ub1*Va0-Ua1*Vb0) +2(cos(O)-1)*(Ua1*Va0+Ub1*Vb0)

# Potential term on y bond 
V2_exact = 2*(Fv[s0,s2]-F0[s0,s2])
V2_est = -2*sin(O)*(Ub0*Va2-Ua0*Vb2) +2(cos(O)-1)*(Ua0*Va2+Ub0*Vb2)

# Expectation values of H0 
H0_exp_exact = tr(-M0*(Fv'-F0'))
H0_exp_est = (1-cos(O))*(a'*U0'*M0*V0*a+b'*U0'*M0*V0*b) 

E2_est = 4*F0[s1,s0] + H0_exp_est + V1_est + V2_est
E2_exact = tr(-Mv*Fv'+M0*F0')