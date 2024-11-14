import OptimalTransport
import numpy as np
import intersection
import ForwardSelection

def interval_DA(ns, nt, X_, B, S_, h_, a, b):
    Bc = np.delete(np.array(range(ns*nt)), B)

    OMEGA = OptimalTransport.constructOMEGA(ns,nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X_.shape[1]-1):
        c_ += (OMEGA.dot(X_[:, [i]])) * (OMEGA.dot(X_[:, [i]]))

    Omega_a = OMEGA.dot(a)
    Omega_b = OMEGA.dot(b)

    w_tilde = c_ + Omega_a * Omega_a
    r_tilde = Omega_a * Omega_b + Omega_b * Omega_a
    o_tilde = Omega_b * Omega_b
    S_B_invS_Bc = np.linalg.inv(S_[:, B]).dot(S_[:, Bc])

    w = (w_tilde[Bc, :].T - w_tilde[B, :].T.dot(S_B_invS_Bc)).T
    r = (r_tilde[Bc, :].T - r_tilde[B, :].T.dot(S_B_invS_Bc)).T
    o = (o_tilde[Bc, :].T - o_tilde[B, :].T.dot(S_B_invS_Bc)).T
    list_intervals = []

    interval = [(-np.inf, np.inf)]
    for i in range(w.shape[0]):
        g3 = - o[i][0]
        g2 = - r[i][0]
        g1 = - w[i][0]
        itv = intersection.solve_quadratic_inequality(g3,g2,g1)

        interval = intersection.Intersec(interval, itv)
    return interval

def interval_SFS(X, Y, K, lst_SELEC_k, lst_Portho, a, b):
    n_sample, n_fea = X.shape

    A=[]
    B=[]

    I = np.identity(n_sample)   

    for step in range(1, K+1):
        P_pp_Mk_1 = lst_Portho[step - 1]
        Xjk = X[:, [lst_SELEC_k[step][-1]]].copy()
        sign_projk = np.sign(np.dot(Xjk.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
        
        projk = sign_projk*(np.dot(Xjk.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xjk))

        if step == 1:
            A.append(-1*projk[0].copy())
            B.append(0)
        for otherfea in range(n_fea):
            if otherfea not in lst_SELEC_k[step]:

                Xj = X[:, [otherfea]].copy()
                sign_proj = np.sign(np.dot(Xj.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
                proj = sign_proj*(np.dot(Xj.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xj))

                A.append(-1*(projk-proj)[0].copy())
                B.append(0)
                A.append(-1*(projk+proj)[0].copy())
                B.append(0)


    A = np.array(A)
    B = np.array(B).reshape((-1,1))

    Ac = np.dot(A,  b)
    Az = np.dot(A,  a)

    Vminus = np.NINF
    Vplus = np.inf

    for j in range(len(B)):
        left = Ac[j][0]
        right = B[j][0] - Az[j][0]

        if abs(right) < 1e-14:
            right = 0
        if abs(left) < 1e-14:
            left = 0
        
        if left == 0:
            if right < 0:
                print('Error')
        else:
            temp = right / left
            if left > 0: 
                Vplus = min(Vplus, temp)
            else:
                Vminus = max(Vminus, temp)
    return Vminus, Vplus
    # return np.around(Vminus, 10), np.around(Vplus, 10)

def differen(a, b):
    c1 = None
    c2 = None
    for i in a:
        if i not in b:
            c1 = i
    for i in b:
        if i not in a:
            c2 = i
    return c1, c2
def interval_SBS(X, Y, K, lst_SELEC_k, a, b):
    n_sample, n_fea = X.shape
    if K == n_fea:
        return -np.inf, np.inf
    A=[]
    B=[]
    I = np.identity(n_sample)  
     
    for step in range(1, n_fea - K + 1):
        Mk = lst_SELEC_k[step]
        Mk_1 = lst_SELEC_k[step - 1]
        # print(f"Mk: {Mk}")
        # print(f"Mk-1: {Mk_1}")
        for each in Mk_1:
            Mj = [x for x in Mk_1 if x != each]
            # print(f"----Mj: {Mj}")
            jk, j = differen(Mk, Mj)
            if jk == None and j == None:
                continue
            same = [x for x in Mk_1 if x != jk and x != j]
            # print('-----------',jk, j)
            # print('-------same',same)

            Xsame = X[:, same].copy()
            P_pp_Mk_1 = I - np.dot(np.dot(Xsame, np.linalg.inv(np.dot(Xsame.T, Xsame))), Xsame.T)
            Xjk = X[:, [jk]].copy()
            Xj = X[:, [j]].copy()

            sign_projk = np.sign(np.dot(Xjk.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
            projk = sign_projk*(np.dot(Xjk.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xjk))
            # if step == 1:
            # A.append(-1*projk[0].copy())
            # B.append(0)

            sign_proj = np.sign(np.dot(Xj.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
            proj = sign_proj*(np.dot(Xj.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xj))

            A.append(-1*(projk-proj)[0].copy())
            B.append(0)
            A.append(-1*(projk+proj)[0].copy())
            B.append(0)

    A = np.array(A)
    B = np.array(B).reshape((-1,1))

    Ac = np.dot(A,  b)
    Az = np.dot(A,  a)

    Vminus = np.NINF
    Vplus = np.inf

    for j in range(len(B)):
        left = Ac[j][0]
        right = B[j][0] - Az[j][0]

        if abs(right) < 1e-14:
            right = 0
        if abs(left) < 1e-14:
            left = 0
        
        if left == 0:
            if right < 0:
                print('Error')
        else:
            temp = right / left
            if left > 0: 
                Vplus = min(Vplus, temp)
            else:
                Vminus = max(Vminus, temp)
    return Vminus, Vplus

def interval_AIC_BS(X, Y, Portho, K, a, b, Sigma, seed = 0):
    n_sample, n_fea = X.shape

    A = []
    Pka = Portho[K-1].dot(a) 
    Pkb = Portho[K-1].dot(b)

    intervals = [(-np.inf, np.inf)] 

    for step in range(0, n_fea):
        if step != K-1:
            Pja = Portho[step].dot(a)
            Pjb = Portho[step].dot(b)
            g1 = Pka.T.dot(Sigma.dot(Pka)) - Pja.T.dot(Sigma.dot(Pja)) + 2*(K - step-1)
            g2 = Pka.T.dot(Sigma.dot(Pkb)) + Pkb.T.dot(Sigma.dot(Pka)) - Pja.T.dot(Sigma.dot(Pjb)) - Pjb.T.dot(Sigma.dot(Pja))
            g3 = Pkb.T.dot(Sigma.dot(Pkb)) - Pjb.T.dot(Sigma.dot(Pjb))

            g1, g2, g3 = g1.item(), g2.item(), g3.item()

            itv = intersection.solve_quadratic_inequality(g3, g2, g1, seed)

            intervals = intersection.Intersec(intervals, itv)
    return intervals 

def OC_DA_BS_AIC(ns, nt, a, b, XsXt_, Xtilde, Ytilde, Sigmatilde, B, S_, h_, SELECTION_F, GAMMA,seed = 0):

    lst_SELECk, lst_P = ForwardSelection.list_residualvec_BS(Xtilde, Ytilde)
    lst_SELECk.reverse()
    itvDA = interval_DA(ns, nt, XsXt_, B, S_, h_, a, b)
    itvBS = [interval_SBS(Xtilde, Ytilde, 
                                    len(SELECTION_F),
                                    lst_SELECk,
                                    GAMMA.dot(a), GAMMA.dot(b))]
    itvAIC = interval_AIC_BS(Xtilde, Ytilde, 
                                        lst_P, len(SELECTION_F), 
                                        GAMMA.dot(a), GAMMA.dot(b), Sigmatilde, seed)

    finalinterval = intersection.Intersec(itvDA, itvBS) 
    finalinterval = intersection.Intersec(finalinterval, itvAIC)
    # print(f"da: {itvDA} | fs: {itvBS} | aic: {itvAIC}")
    return finalinterval


def OC_fixedBS_interval(ns, nt, a, b, XsXt_, Xtilde, Ytilde, Sigmatilde, B, S_, h_, SELECTION_F, GAMMA,):

    lst_SELECk = ForwardSelection.list_residualvec_BS(Xtilde, Ytilde)[0]
    lst_SELECk.reverse()
    itvDA = interval_DA(ns, nt, XsXt_, B, S_, h_, a, b)
    itvBS = [interval_SBS(Xtilde, Ytilde, len(SELECTION_F),
                                    lst_SELECk, GAMMA.dot(a), GAMMA.dot(b))]
    # print(itvDA, itvFS)
    finalinterval = intersection.Intersec(itvDA, itvBS) 
    return finalinterval, itvDA, itvBS