# This file contains calculations for hopping of vison pairs due to a heisenberg interaction in the monolayer

L1 = 15
L2 = 15
m = 0
BCs = [L1,L2,m]

function calculate_heisenberg_hopping_with_second_pair(BCs,R_sep,flavour="z",K=1)
    """
    R_sep is the seperation [n1,n2] of the 2nd static vison pair with respect to the first 
    K = 1 for FM Kitaev
    K = -1 for AFM Kitaev 
    """

    M0 = K*get_M0(BCs)

    Mb = flip_bond_variable(M0,BCs,R_sep,flavour)
    M1 = flip_bond_variable(Mb,BCs,[0,0],"z")
    M2 = flip_bond_variable(Mb,BCs,[1,0],"z")
    M10 = flip_bond_variable(M0,BCs,[0,0],"z")
    M20 = flip_bond_variable(M0,BCs,[1,0],"z")

    U1 ,V1 = get_U_and_V(M1)
    U2 ,V2 = get_U_and_V(M2)

    U10 ,V10 = get_U_and_V(M10)
    U20 ,V20 = get_U_and_V(M20)

    U = U2'*U1
    V = V2'*V1

    U_0 = U20'*U10
    V_0 = V20'*V10

    X0 = 0.5*(U_0+V_0)
    Y0 = 0.5*(U_0-V_0)

    X = 0.5*(U+V)
    Y = 0.5*(U-V)

    Z = inv(X)*Y
    Z = 0.5*(Z-Z')
    Z0 = inv(X0)*Y0
    Z0 = 0.5*(Z0-Z0')
    display(svd(Z0).S)

    C0 = abs(det(X0))^(0.5)
    C = abs(det(X))^(0.5)

    display(C0)
    display(C0*(-(U10*V10')[2,1]))
    display(C0*(-(U10*V10')[2,1]+(U10*Z0*V10')[2,1]))

    hop_0 = C0*(1-(U10*V10')[2,1]+(U10*Z0*V10')[2,1])
    hop = C*(1-(U1*V1')[2,1]+(U1*Z*V1')[2,1])

    display(hop_0)
    display(hop)

end 

function calculate_heisenberg_hopping_with_single_vison_background(BCs,n2,R_sep,K=1)
    """
    R_sep is the seperation [n1,n2] of the 2nd static vison pair with respect to the first 
    K = 1 for FM Kitaev
    K = -1 for AFM Kitaev 
    """

    M0 = K*get_M0(BCs)

    Mb = M0
    for n1 = 1:R_sep 
        Mb = flip_bond_variable(Mb,BCs,[n1,n2],"y")
    end

    plot_bond_energies_2D(Mb,BCs)

    M1 = flip_bond_variable(Mb,BCs,[0,0],"z")
    M2 = flip_bond_variable(Mb,BCs,[1,0],"z")
    M10 = flip_bond_variable(M0,BCs,[0,0],"z")
    M20 = flip_bond_variable(M0,BCs,[1,0],"z")

    U1 ,V1 = get_U_and_V(M1)
    U2 ,V2 = get_U_and_V(M2)

    U10 ,V10 = get_U_and_V(M10)
    U20 ,V20 = get_U_and_V(M20)

    U = U2'*U1
    V = V2'*V1

    U_0 = U20'*U10
    V_0 = V20'*V10

    X0 = 0.5*(U_0+V_0)
    Y0 = 0.5*(U_0-V_0)

    X = 0.5*(U+V)
    Y = 0.5*(U-V)

    Z = inv(X)*Y
    Z = 0.5*(Z-Z')
    Z0 = inv(X0)*Y0
    Z0 = 0.5*(Z0-Z0')

    C0 = abs(det(X0))^(0.5)
    C = abs(det(X))^(0.5)

    hop_0 = C0*(1-(U10*V10')[2,1]+(U10*Z0*V10')[2,1])
    hop = C*(1-(U1*V1')[2,1]+(U1*Z*V1')[2,1])

    display(C0)

    display(hop_0)
    display(hop)
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
    #display(Pfaffian(V12*U12'-(V12*U12')'))

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
    #display(abs(det(X10))*(Pfaffian([0.5*(M20-M20') -I ; I -0.5*(M10-M10') ])))

    #display(pfaffian([0.5*(M20-M20') -I ; I -0.5*(M10-M10') ]))
    #display(det([0.5*(M20-M20') -I ; I -0.5*(M10-M10') ])^(0.5))
    
    return hop
end 

function get_M_eff(BCs)
    N = BCs[1]*BCs[2]

    M0 = get_M0(BCs)

    M_eff = zeros(N,N)
    for j = 1:N
        Mv = get_M0(BCs)
        Mv[j,j] = -1

        M_eff += sqrt(Mv*Mv')/N
        display(j)
    end 

    return M_eff
end