import json

import numpy as np

from waterimg_class import *

from diff_attack import *

all_msg_tot1, all_msg_same1, all_inv_tot1, all_inv_same1 = 0,0,0,0
all_msg_tot3, all_msg_succ3, all_inv_tot3, all_inv_succ3 = 0, 0, 0, 0

all_msg_tot2, all_msg_same2, all_inv_tot2, all_inv_same2 = 0, 0, 0, 0
all_msg_tot4, all_msg_succ4, all_inv_tot4, all_inv_succ4 = 0, 0, 0, 0

all_key_plain_tot, all_key_plain_detect = 0, 0
all_dup_plain_tot, all_dup_plain_detect = 0, 0
all_key_plain_sum, all_key_plain_same = 0, 0
all_dup_plain_sum, all_dup_plain_same = 0, 0

threshold = 0.75

dup_num_tot = 0
error_num_tot = 0

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def print_json_keys(file_path,t):
    global all_msg_tot1, all_msg_same1, all_inv_tot1, all_inv_same1
    global all_msg_tot3, all_msg_succ3, all_inv_tot3, all_inv_succ3
    global all_msg_tot2, all_msg_same2, all_inv_tot2, all_inv_same2
    global all_msg_tot4, all_msg_succ4, all_inv_tot4, all_inv_succ4
    global dup_num_tot, error_num_tot
    global all_key_plain_tot, all_key_plain_detect, all_dup_plain_tot, all_dup_plain_detect
    global all_key_plain_sum, all_key_plain_same, all_dup_plain_sum, all_dup_plain_same

    prc_codeword_list = []
    inv_codeword_list = []
    secret_key = []
    secret_key_dict = {}
    public_key_trans = []
    public_key = []
    otp = []

    check_dup = {}
    dup1, dup2 = -1,-1

    wim = None




    try:
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)



        
        
        
        

        
        if 'secret_key' in data:
            
            
            
            
            
            secret_key = data['secret_key']
            for i in range(len(secret_key)):
                for j in range(len(secret_key[0])):
                    secret_key[i][j] = int(secret_key[i][j])
            for ele in secret_key:
                if tuple(sorted(ele)) not in secret_key_dict:
                    secret_key_dict[tuple(sorted(ele))] = 1
            

        
        if 'public_key' in data:
            public_key = data['public_key']
            n, k, t = len(public_key), len(public_key[0]), t
            public_key_trans = [[0 for _ in range(n)] for _ in range(k)]
            
            for i in range(len(public_key)):
                for j in range(len(public_key[i])):
                    public_key[i][j] = int(public_key[i][j])
                    public_key_trans[j][i] = (public_key[i][j])
            check_dup = find_duplicate_rows(public_key)


        
        if 'one_time_pad' in data:
            otp = data['one_time_pad']



        if 'prc_codeword_save' in data:
            
            
            prc_codeword_list = data['prc_codeword_save']
            for i in range(len(prc_codeword_list)):
                for j in range(len(prc_codeword_list[i])):
                    prc_codeword_list[i][j] = int(prc_codeword_list[i][j])

        if 'PRC_reverse_code' in data:
            
            
            
            inv_codeword_list = data['PRC_reverse_code']
            for i in range(len(inv_codeword_list)):
                for j in range(len(prc_codeword_list[i])):
                    inv_codeword_list[i][j] = int(inv_codeword_list[i][j])


        wim = WaterImg(t=t,public_key=public_key,otp=otp,
                       origin_msg_list=prc_codeword_list,inv_msg_list=inv_codeword_list)
        avg_err_num = wim.avg_err_num()
        error_num_tot += avg_err_num
        print(f"avg_err_num={avg_err_num}")

        msg_tot1,msg_same1,inv_tot1,inv_same1,dup_plain_tot,dup_plain_same,dup_num = wim.dup_detect()
        msg_tot2,msg_same2,inv_tot2,inv_same2,key_plain_tot,key_plain_same = wim.part_key_recover()

        dup_num_tot += dup_num
        all_msg_tot1, all_msg_same1, all_inv_tot1, all_inv_same1 = \
            all_msg_tot1+msg_tot1, all_msg_same1+msg_same1, all_inv_tot1+inv_tot1, all_inv_same1+inv_same1

        all_msg_tot2, all_msg_same2, all_inv_tot2, all_inv_same2 = \
            all_msg_tot2 + msg_tot2, all_msg_same2 + msg_same2, all_inv_tot2 + inv_tot2, all_inv_same2 + inv_same2

        all_msg_tot3 += 1
        if msg_tot1 == 0:
            all_msg_tot3 -= 1
        elif msg_same1 / msg_tot1 >= threshold:
            all_msg_succ3 += 1

        all_inv_tot3 += 1
        if inv_tot1 == 0:
            all_inv_tot3 -= 1
        elif inv_same1 / inv_tot1 >= threshold:
            all_inv_succ3 += 1

        all_msg_tot4 += 1
        if msg_tot2 == 0:
            all_msg_tot4 -= 1
        elif msg_same2 / msg_tot2 >= threshold:
            all_msg_succ4 += 1
        all_inv_tot4 += 1

        if inv_tot2 == 0:
            all_inv_tot4 -= 1
        elif inv_same2 / inv_tot2 >= threshold:
            all_inv_succ4 += 1

        all_key_plain_sum += key_plain_tot
        all_key_plain_same += key_plain_same
        if key_plain_tot > 0:
            all_key_plain_tot += 1
            if key_plain_same / key_plain_tot >= threshold:
                all_key_plain_detect += 1

        all_dup_plain_sum += dup_plain_tot
        all_dup_plain_same += dup_plain_same
        if dup_plain_tot > 0:
            all_dup_plain_tot += 1
            if dup_plain_same / dup_plain_tot >= threshold:
                all_dup_plain_detect += 1


        
        
        
        
        
        
        

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except json.JSONDecodeError:
        print("错误：文件内容不是有效的JSON格式")
    except Exception as e:
        print(f"发生错误：{str(e)}")

def safe_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else float('NaN')


if __name__ == "__main__":
    num = 128
    for i in range(1,num+1):
        file_path = "../gim_data/gim_normal_t3/"+ str(i).zfill(4) +".json"
        print_json_keys(file_path,t=3)
        print(f"avg_err_num={error_num_tot/i}")
        print(f"===dup===")
        print(f"dup_num_tot={dup_num_tot}")
        print(f"All:    {all_msg_tot1}, {all_msg_same1}, {all_inv_tot1}, {all_inv_same1}")
        print(f"Detect: {all_msg_tot3}, {all_msg_succ3}, {all_inv_tot3}, {all_inv_succ3}")
        print(f"Plain All: {all_dup_plain_sum}, {all_dup_plain_same}")
        print(f"Plain Detect:  {all_dup_plain_tot}, {all_dup_plain_detect}")
        print(f"===key===")
        print(f"All:    {all_msg_tot2}, {all_msg_same2}, {all_inv_tot2}, {all_inv_same2}")
        print(f"Detect: {all_msg_tot4}, {all_msg_succ4}, {all_inv_tot4}, {all_inv_succ4}")
        print(f"Plain All: {all_key_plain_sum}, {all_key_plain_same}")
        print(f"Plain Detect:  {all_key_plain_tot}, {all_key_plain_detect}")

        print("===Summary===")

        
        TPR_0 = safe_div(all_msg_succ4, all_msg_tot4)
        TPR_1 = safe_div(all_inv_succ4, all_inv_tot4)
        FPR = safe_div(all_key_plain_detect, all_key_plain_tot)

        print(f"Attack I: TPR_0:{TPR_0} TPR_1:{TPR_1} FPR:{FPR} Success Rate:{all_msg_tot4/i}")

        
        TPR_0 = safe_div(all_msg_succ3, all_msg_tot3)
        TPR_1 = safe_div(all_inv_succ3, all_inv_tot3)
        FPR = safe_div(all_dup_plain_detect, all_dup_plain_tot)

        print(f"Attack II: TPR_0:{TPR_0} TPR_1:{TPR_1} FPR:{FPR} Success Rate:{all_msg_tot3/i}")