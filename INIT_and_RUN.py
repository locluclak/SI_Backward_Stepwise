import pivot
import pivot_nonDA
import numpy as np
import time 
def run(iter = 0):    
    seed = int(np.random.rand() * (2**32 - 1))
    # seed = 939590937   

    # print("Seed:",seed)

    # st = time.time()
    #___________________________________________________________
    ns = 100
    nt = 20
    p = 5
    betat = 0
    true_beta_s = np.full((p,1), 2) #source's beta
    true_beta_t = np.full((p,1), betat) #target's beta
    k = 3 # k = -1 if choose AIC
    #___________________________________________________________
    # en = time.time()
    pvalue = pivot.pvalue_SI(seed, ns, nt, p, true_beta_s, true_beta_t, k)

    # pvalue = pivot_nonDA.pvalue_SI(seed, ns, p, true_beta_t)

    # Save pvalue into file
    OCorPARA_FIXorAIC_FPRorTPR ='para_correlated_fixed'
    # filename = f'Experiment/LstpBS_{OCorPARA_FIXorAIC_FPRorTPR}_{ns}_{p}.txt'
    filename = f'Experiment/LstBS_{OCorPARA_FIXorAIC_FPRorTPR}_{ns}_{p}_{betat}.txt'
    with open(filename, 'a') as f:
        f.write(str(pvalue)+ '\n')
    return pvalue

if __name__ == "__main__":
    for i in range(1):
        # st = time.time()
        print(run())
        # en = time.time()
        # print(f"Time of 1 pvalue {i}: {en - st}")s