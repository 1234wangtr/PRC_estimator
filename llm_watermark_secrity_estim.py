
# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    getcontext().Emax = 999999  # 调大最大指数
    getcontext().Emin = -999999  # 调小最小指数

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
        # 计算 n! / (n-m)! / n^m
        return exact_perm_prob(n,m)
        #return math.perm(n, m) / (n ** m)

def get_g_dup_prob(n,r,t):
    N = int(calc_C(n - r, t - 1))
    M = r
    res = probability_no_duplicates(N, M)
    # print(f"res={res} N={N} M={M}")
    dup = 1-res
    return dup


def main():

    logr = 17
    r = int(2**logr)

    plt.figure(figsize=(6, 3.5))
    n = 2 ** 14

    isd_list = []
    mitm_list = []
    partial_list = []
    dup_list = []

    t_values = [3,4,5,6,7,8,9,10,11,12,13]

    for t in t_values:
        n = r  # * 2
        r = int(0.99 * n)
        logr = math.log2(r)

        eps = 0.5 / 2**((0.25*logr-1)/t)


        g = int(math.log2(calc_C(n,t)))
        msg2 = (1 - 0.5 + eps) ** (g)
        msg2 = 1 / msg2
        msg2 *= n ** 3
        msg2 = math.log2(msg2)
        print(f"msg2={msg2}")

        key_recovery = math.log2(g*calc_C(n, (t+1)//2))

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
        # print(f"dup={dup}")
        try:
            dup = math.log2(dup)
        except:
            dup = -100000


        print(f"duprob={dup}")
        dup = math.log2(n * g) - dup

        print(f"{t}\t{eps}\t{0.5-eps}\t{key_recovery}\t{distinguish}\t{msg2}\t{dup}")
        isd_list.append(msg2)
        mitm_list.append(key_recovery)
        partial_list.append(distinguish)
        dup_list.append(dup)

    plt.plot(t_values, isd_list, marker='o', label=r'$\text{T}_{\mathsf{ISD}}$')
    plt.plot(t_values, mitm_list, marker='s', label=r'$\text{T}_{\mathsf{MITM}}$')
    plt.plot(t_values, partial_list, marker='^', label=r'$\text{T}_{\mathsf{partial}}$')
    plt.plot(t_values, dup_list, marker='d', label=r'$\text{T}_{\mathsf{dis}}$')

    # 添加标题和标签
    # plt.title('Time Complexity Comparison', fontsize=14)
    plt.xlabel('$t$', fontsize=12)
    plt.ylabel('Time Complexity', fontsize=12)

    # 添加图例和网格
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    # 显示图形
    # plt.show()
    plt.savefig('llm_watermark_secrity_estim_origin.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()

