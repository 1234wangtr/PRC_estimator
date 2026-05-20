# %%
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme(style="whitegrid", font="Times New Roman",font_scale=1.4)
def calc_C(big,small):
    sum = 1
    if big<=small:
        return 1
    i = 0
    while i < small:
        sum *= big-i
        i += 1
    i = 1
    while i <= small:
        sum //= i
        i += 1
    return sum


from decimal import Decimal, getcontext


def exact_perm_prob(n, m, precision=60):
    getcontext().prec = precision
    getcontext().Emax = 999999
    getcontext().Emin = -999999

    result = Decimal(1)
    for i in range(m):
        term = Decimal(n - i) / Decimal(n)
        result *= term

    return result


def probability_no_duplicates(n, m):
    if m > n:
        return 0.0
    elif m == 1:
        return 1.0
    else:
        return exact_perm_prob(n,m)

def get_g_dup_prob(n,r,t):
    N = int(calc_C(n - r, t - 1))
    M = r
    res = probability_no_duplicates(N, M)
    dup = 1-res
    return dup


def main():

    logn = 17
    #r = int(2**logr)
    n = int(2**logn)
    plt.figure(figsize=(5,3.5))

    isd_list = []
    partial_list = []
    dup_list = []
    lambda_list = []
    eps_list = []
    rho_list = []
    dup_prob_list= []
    t_values = [3,4,5,6,7,8,9,10,11,12,13,14]

    for t in t_values:
        r = int(0.99 * n)
        logr = math.log2(r)

        eps = 0.5 / 2**((0.25*logr-1)/t)
        eps_list.append(eps)


        g = int(math.log2(calc_C(n,t)))
        msg2 = (1 - 0.5 + eps) ** (g)
        msg2 = 1 / msg2
        msg2 *= n ** 3
        msg2 = math.log2(msg2)
        print(f"msg2={msg2}")


        # recover one
        if t%2==0:
            pr = calc_C(n//2,t//2)*calc_C(n//2,t//2) / calc_C(n,t)
            num = int(r*pr)
            distinguish = math.log2( g*calc_C(n//2,t//2)/math.sqrt(num) )
        else:
            pr = calc_C(n // 2, (t+1) // 2) * calc_C(n // 2, t // 2) / calc_C(n, t)
            num = int(r * pr)
            alpha = math.sqrt( (t+1)//2 /  (n//2-t//2)  / num )
            res = calc_C(n//2,(t+1)//2) * alpha
            distinguish = math.log2(g * res)


        # G dup
        dup = get_g_dup_prob(n,r,t)
        try:
            dup = math.log2(dup)
        except:
            dup = -100000

        dup_prob_list.append(dup)
        print(f"duprob={dup}")
        
        dup = math.log2(n * g) - dup

        tmp_lambda = math.log2(calc_C(n,t))
        rho = 0.5 - eps
        rho_list.append(rho)
        print(f"{t}\t{eps}\t{rho}\t{distinguish}\t{msg2}\t{dup}\t{tmp_lambda}")
        isd_list.append(msg2)
        partial_list.append(distinguish)
        dup_list.append(dup)
        lambda_list.append(tmp_lambda)

    plt.plot(t_values, isd_list, marker='o', label=r'$\text{T}_{\mathsf{overlay}}$')
    plt.plot(t_values, partial_list, marker='^', label=r'$\text{T}_{\mathsf{partial}}$')
    plt.plot(t_values, dup_list, marker='d', label=r'$\text{T}_{\mathsf{dis}}$')
    plt.plot(t_values, lambda_list, linestyle='--',color='purple', label=r'$\lambda$')

    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('Complexities (bit)', fontsize=20)

    plt.legend(fontsize=13, loc='best', handlelength=0.8)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xticks(range(int(min(t_values)), int(max(t_values)) + 1), fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig('llm_watermark_secrity_estim.pdf', bbox_inches='tight')
    
    plt.show()

    df = pd.DataFrame({
        't': t_values,
        'eps': eps_list,
        'rho': rho_list,
        'T_partial': partial_list,
        'P_weak': dup_prob_list,
        'T_dis': dup_list,
        'T_overlay': isd_list,
        'lambda': lambda_list
    })
    df["eps"] = df["eps"].round(3)
    df["rho"] = df["rho"].round(3)
    # round T_* to 2 decimal places
    df['T_partial'] = df['T_partial'].round(2)
    df['P_weak'] = df['P_weak'].round(2)
    df['T_dis'] = df['T_dis'].round(2)
    df['T_overlay'] = df['T_overlay'].round(2)
    df['lambda'] = df['lambda'].round(0)
    df.to_csv('llm_watermark_secrity_estim.csv', index=False)

if __name__ == '__main__':
    main()


# %%

# %%
