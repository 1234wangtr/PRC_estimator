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


logr = np.arange(14,15)


def calculate_epsilon(logr, t):
    print(f"calc eps:logr={logr} t={t}")
    return 2 ** ((-0.25 * logr + 1) / t - 1)


t_values = [3, 4, 5,6,7]




def main2():
    # 创建画布
    plt.figure(figsize=(5,3.5))
    n = 2**14

    isd_list = []
    mitm_list = []
    partial_list = []
    dup_list = []
    true_isd_list = []

    for t in t_values:
        message_length = 512
        false_positive_rate = 1e-5
        num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
        #secpar = int(np.log2(calc_C(n, t)))
        g = int(math.log2(calc_C(n, t)))
        secpar = g
        k = message_length + g + num_test_bits
        r = n - k - secpar

        logr = math.log2(r)
        eps = calculate_epsilon(logr, t)

        noise_rate = 1 - 2 ** (-secpar / g ** 2)
        print(f"logr={logr} t={t} n={n} r={r} eps={eps} g={g} k={k} noise_rate={noise_rate}")
        # eps = 0.5-noise_rate

        # mu = int(n * (0.5 - eps))
        # for i in range(mu):
        #     res *= (n - k - mu) / (n - mu)
        # res = n ** 3 / res
        # res = math.log2(res)
        # msg = res

        msg2 = (1 - 0.5 + eps) ** (k)
        msg2 = 1 / msg2
        msg2 *= n ** 3
        msg2 = math.log2(msg2)

        no_dup = 1
        for i in range(r):
            no_dup *= (1 - (i) / (calc_C(n - r + i, t - 1)))
            i += 1
        dup_comp = math.log2(n * k) - math.log2(1 - no_dup)
        print(f"dup={math.log2(1 - no_dup)} dup_comp={dup_comp}")


        if t % 2 == 0:
            pr = calc_C(n // 2, t // 2) * calc_C(n // 2, t // 2) / calc_C(n, t)
            num = int(r * pr)
            distinguish = math.log2(k * calc_C(n // 2, t // 2) / math.sqrt(num))
        else:
            pr = calc_C(n // 2, (t + 1) // 2) * calc_C(n // 2, t // 2) / calc_C(n, t)
            num = int(r * pr)
            alpha = math.sqrt((t + 1) // 2 / (n // 2 - t // 2) / num)
            res = calc_C(n // 2, (t + 1) // 2) * alpha
            distinguish = math.log2(k * res)

        MITM = math.log2(k * calc_C(n, (t + 1) // 2))

        partial = distinguish
        isd = msg2

        msg3 = (1 - noise_rate) ** (k)
        msg3 = 1 / msg3
        msg3 *= n ** 3
        msg3 = math.log2(msg3)


        print(f"isd={isd} true_isd={msg3} MITM={MITM} partial={partial} dup={dup_comp}")
        isd_list.append(isd)
        mitm_list.append(MITM)
        partial_list.append(partial)
        dup_list.append(dup_comp)
        true_isd_list.append(msg3)

        i += 1


    # 绘制四条折线图
    plt.plot(t_values, isd_list, marker='o', label=r'${\text{T}_{\mathsf{eo}}}$')
    plt.plot(t_values, true_isd_list, marker='*', label=r"$\text{T}_{\mathsf{eo'}}$")
    plt.plot(t_values, mitm_list, marker='s', label=r'$\text{T}_{\mathsf{equiv}}$')
    plt.plot(t_values, partial_list, marker='^', label=r'$\text{T}_{\mathsf{partial}}$')
    plt.plot(t_values, dup_list, marker='d', label=r'$\text{T}_{\mathsf{dis}}$')

    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('Complexities (bit)', fontsize=20)

    # 添加图例和网格
    plt.legend(fontsize=13, loc='best', handlelength=0.8)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xticks(range(int(min(t_values)), int(max(t_values)) + 1), fontsize=20)
    plt.yticks(fontsize=20)
    # 显示图形
    # plt.show()
    # plt.savefig('gim_watermark_secrity_estim.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main2()