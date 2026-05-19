import numpy as np
import itertools
from math import comb
from collections import defaultdict

import random

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



def find_duplicate_rows(matrix):
    """
    高效检测 0-1 矩阵中的重复行，并返回重复行的索引。

    参数:
        matrix: List[List[int]] -- 输入的 n×k 0-1 矩阵

    返回:
        Dict[Tuple[int, ...], List[int]] -- 键是重复的行内容，值是对应的行索引列表
                                          如果没有重复行，返回空字典。
    """
    row_dict = {}
    duplicates = {}

    for idx, row in enumerate(matrix):
        # 将行转换为元组（可哈希）
        row_tuple = tuple(row)

        if row_tuple in row_dict:
            # 如果该行已存在，记录重复
            if row_tuple not in duplicates:
                duplicates[row_tuple] = [row_dict[row_tuple]]
            duplicates[row_tuple].append(idx)
        else:
            # 否则存储当前行和索引
            row_dict[row_tuple] = idx

    return duplicates


def generate_constrained_vectors(n, weight, x=-1, position='front'):
    """
    生成 x 个随机的 n 位 0-1 向量（含 weight 个 1），且所有 1 位于前 n/2 或后 n/2 位

    参数:
        n: 向量总长度
        weight: 向量中 1 的数量
        x: 生成的向量数量（默认 -1 表示生成所有可能组合）
        position: 'front'（1在前半段）或 'back'（1在后半段）

    返回:
        List[tuple]: 每个 tuple 是 1 的位置索引组合
    """
    half = n // 2

    if position == 'front':
        valid_range = range(half)  # 1 只能在前半段 [0, n/2)
    elif position == 'back':
        valid_range = range(half, n)  # 1 只能在后半段 [n/2, n)
    else:
        raise ValueError("position 必须是 'front' 或 'back'")

    # 获取所有可能的有效组合
    all_possible = list(itertools.combinations(valid_range, weight))

    # 随机选择 x 个组合（若 x=-1 返回全部）
    if x == -1:
        return all_possible
    else:
        return random.sample(all_possible, min(x, len(all_possible)))


def generate_random_vectors(n, weight, x=-1):
    """生成 x 个随机的 n 位 0-1 向量（含 weight 个 1）"""
    # all_possible = itertools.combinations(range(n), weight)  # 所有可能的 1 的位置组合
    sampled_indices = random.sample(list(itertools.combinations(range(n), weight)), x)  # 随机选 x 个
    return sampled_indices

def part_key_recovery(n,k,t,G,secret_key_dict={}):
    """Meet-in-the-middle algorithm for finding vectors in F₂^n"""
    print(f"part key recovery")
    half_t = (t) // 2
    remaining_t = t - half_t


    # Generate L1 and L2 sets (only positions of 1s)
    num = calc_C(n,half_t)
    num = 500000
    num = 500000
    if t==3:
        num = calc_C(n,half_t)
        L1 = list(generate_random_vectors(n, half_t, num))
        L2 = list(generate_random_vectors(n, remaining_t, num))
    elif t==4:
        num = 500000
        L1 = list(generate_constrained_vectors(n, half_t, num, 'front'))
        L2 = list(generate_constrained_vectors(n, half_t, num, 'back'))

    print(f"initial over")




    # For demonstration, let's create a random target matrix G (k x n)
    # In a real implementation, you would use your actual G matrix

    # G = [[random.randint(0, 1) for _ in range(n)] for _ in range(k)]
    # for row in range(k):
    #     print(G[row])

    # Precompute syndromes for L1
    L1_full = {}

    for vec in L1:
        # Compute syndrome: G * vec (in F₂)
        syndrome = [0] * k
        for pos in vec:
            for i in range(k):
                syndrome[i] ^= G[i][pos]
        # Convert syndrome to a tuple for hashing
        syndrome_tuple = tuple(syndrome)
        if syndrome_tuple not in L1_full:
            L1_full[syndrome_tuple] = []
        L1_full[syndrome_tuple].append(vec)

    # print(L1_full)
    # Search for matches in L2
    solutions = {}

    for vec in L2:
        # Compute target syndrome for this vector
        syndrome = [0] * k
        for pos in vec:
            for i in range(k):
                syndrome[i] ^= G[i][pos]
        # We want vectors in L1 that have the same syndrome
        target_syndrome = tuple(syndrome)
        if target_syndrome in L1_full:
            for l1_vec in L1_full[target_syndrome]:
                # Combine the vectors (positions of 1s)
                combined = sorted(set(l1_vec) | set(vec))
                if len(combined) >= 3:
                    print(f"found l1_vec={l1_vec} vec={vec} combined={combined}")
                else:
                    continue
                tmp = tuple(combined)
                # if tmp in secret_key_dict:
                #     print(f"in tmp={tmp}")
                # else:
                #     print(f"out")

                solutions[tmp] = 1
    print(f"found solu num={len(solutions)}")
    return solutions
