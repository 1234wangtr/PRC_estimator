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
    row_dict = {}
    duplicates = {}

    for idx, row in enumerate(matrix):
        
        row_tuple = tuple(row)

        if row_tuple in row_dict:
            
            if row_tuple not in duplicates:
                duplicates[row_tuple] = [row_dict[row_tuple]]
            duplicates[row_tuple].append(idx)
        else:
            
            row_dict[row_tuple] = idx

    return duplicates


def generate_constrained_vectors(n, weight, x=-1, position='front'):
    half = n // 2

    if position == 'front':
        valid_range = range(half)  
    elif position == 'back':
        valid_range = range(half, n)  
    else:
        raise ValueError("position 必须是 'front' 或 'back'")

    
    all_possible = list(itertools.combinations(valid_range, weight))

    
    if x == -1:
        return all_possible
    else:
        return random.sample(all_possible, min(x, len(all_possible)))


def generate_random_vectors(n, weight, x=-1):
    
    sampled_indices = random.sample(list(itertools.combinations(range(n), weight)), x)  
    return sampled_indices

def part_key_recovery(n,k,t,G,secret_key_dict={}):
    print(f"part key recovery")
    half_t = (t) // 2
    remaining_t = t - half_t


    
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




    
    

    
    
    

    
    L1_full = {}

    for vec in L1:
        
        syndrome = [0] * k
        for pos in vec:
            for i in range(k):
                syndrome[i] ^= G[i][pos]
        
        syndrome_tuple = tuple(syndrome)
        if syndrome_tuple not in L1_full:
            L1_full[syndrome_tuple] = []
        L1_full[syndrome_tuple].append(vec)

    
    
    solutions = {}

    for vec in L2:
        
        syndrome = [0] * k
        for pos in vec:
            for i in range(k):
                syndrome[i] ^= G[i][pos]
        
        target_syndrome = tuple(syndrome)
        if target_syndrome in L1_full:
            for l1_vec in L1_full[target_syndrome]:
                
                combined = sorted(set(l1_vec) | set(vec))
                if len(combined) >= 3:
                    print(f"found l1_vec={l1_vec} vec={vec} combined={combined}")
                else:
                    continue
                tmp = tuple(combined)
                
                
                
                

                solutions[tmp] = 1
    print(f"found solu num={len(solutions)}")
    return solutions
