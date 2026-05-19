from diff_attack import *

class WaterImg:
    def __init__(self,t,public_key,otp,origin_msg_list=[],inv_msg_list=[],inv_msg_list_v1_5=[],inv_msg_list_v2=[]):
        self.t = t
        self.secret_key = {}
        self.public_key = public_key    # n x k
        self.otp = otp
        self.origin_msg_list = origin_msg_list
        self.inv_msg_list = inv_msg_list
        self.inv_msg_list_v1_5 = inv_msg_list_v1_5
        self.inv_msg_list_v2 = inv_msg_list_v2
        self.n = len(self.public_key)
        self.k = len(self.public_key[0])
        print(f"k={self.k} n={self.n}")

        for i in range(len(otp)):
            otp[i] = int(otp[i])

        self.plain_msg_list = []
        for i in range(len(self.origin_msg_list)):
            rd = random.choices([0, 1], k=len(self.origin_msg_list[0]))
            self.plain_msg_list.append(rd)
        print(f"dbg len:origin {len(origin_msg_list)} inv {len(inv_msg_list)} inv15 {len(inv_msg_list_v1_5)} inv2 {len(inv_msg_list_v2)}")


    def part_key_recover(self):
        public_key = self.public_key
        n,k, = self.n,self.k
        public_key_trans = [[0 for _ in range(n)] for _ in range(k)]
        # print(f"public key={len(public_key)}")
        for i in range(len(public_key)):
            for j in range(len(public_key[i])):
                public_key_trans[j][i] = (public_key[i][j])
        # full_key_recovery(n, k, t, public_key_trans, secret_key_dict)
        tst_keys = part_key_recovery(self.n,self.k,self.t,public_key_trans)

        msg_tot = 0
        msg_satis = 0
        for i in range(len(self.origin_msg_list)):
            tmp_msg = self.origin_msg_list[i]
            for ele in tst_keys:
                tmp_res = 0
                for j in range(len(ele)):
                    tmp_res += tmp_msg[ele[j]] + self.otp[ele[j]]
                if tmp_res % 2 == 0:
                    msg_satis += 1
                msg_tot += 1
        print(f"msg_tot={msg_tot} msg_satis={msg_satis}")

        inv_tot = 0
        inv_satis = 0
        for i in range(len(self.inv_msg_list)):
            tmp_msg = self.inv_msg_list[i]
            for ele in tst_keys:
                tmp_res = 0
                for j in range(len(ele)):
                    tmp_res += tmp_msg[ele[j]] #+ self.otp[ele[j]]
                if tmp_res % 2 == 0:
                    inv_satis += 1
                inv_tot += 1
        print(f"inv_tot={inv_tot} msg_satis={inv_satis}")

        plain_tot = 0
        plain_satis = 0
        for i in range(len(self.plain_msg_list)):
            tmp_msg = self.plain_msg_list[i]
            for ele in tst_keys:
                tmp_res = 0
                for j in range(len(ele)):
                    tmp_res += tmp_msg[ele[j]]  # + self.otp[ele[j]]
                if tmp_res % 2 == 0:
                    plain_satis += 1
                plain_tot += 1
        print(f"plain_tot={plain_tot} plain_satis={plain_satis}")

        v15_tot = 0
        v15_satis = 0
        for i in range(len(self.inv_msg_list_v1_5)):
            tmp_msg = self.inv_msg_list_v1_5[i]
            for ele in tst_keys:
                tmp_res = 0
                for j in range(len(ele)):
                    tmp_res += tmp_msg[ele[j]]  #+ self.otp[ele[j]]
                if tmp_res % 2 == 0:
                    v15_satis += 1
                v15_tot += 1
        print(f"v15_tot={v15_tot} v15_satis={v15_satis}")

        v2_tot = 0
        v2_satis = 0
        for i in range(len(self.inv_msg_list_v2)):
            tmp_msg = self.inv_msg_list_v2[i]
            for ele in tst_keys:
                tmp_res = 0
                for j in range(len(ele)):
                    tmp_res += tmp_msg[ele[j]]  #+ self.otp[ele[j]]
                if tmp_res % 2 == 0:
                    v2_satis += 1
                v2_tot += 1
        print(f"v2_tot={v2_tot} v2_satis={v2_satis}")

        return msg_tot, msg_satis, inv_tot, inv_satis, plain_tot, plain_satis, v15_tot, v15_satis, v2_tot, v2_satis
        #

    def dup_detect(self):
        dup = find_duplicate_rows(self.public_key)
        dup_dict = {}
        for ele in dup:
            dup_dict[tuple(dup[ele])] = 1
        print(f"dup_dict={dup_dict}")

        # Then distinguish
        msg_tot = 0
        msg_same = 0
        for i in range(len(self.origin_msg_list)):
            tmp_msg = self.origin_msg_list[i]
            for ele in dup_dict:
                if (int(tmp_msg[ele[0]]) + self.otp[ele[0]]  + int(tmp_msg[ele[1]]) + self.otp[ele[1]]) %2 == 0:
                    msg_same += 1
                msg_tot += 1
        print(f"msg_tot={msg_tot} msg_same={msg_same}")

        inv_tot = 0
        inv_same = 0
        for i in range(len(self.inv_msg_list)):
            tmp_msg = self.inv_msg_list[i]
            for ele in dup_dict:
                if (int(tmp_msg[ele[0]])  + int(tmp_msg[ele[1]]) ) % 2 == 0:
                    inv_same += 1
                inv_tot += 1
        print(f"inv_tot={inv_tot} inv_same={inv_same}")
        print(f"len={len(dup_dict)} {dup_dict}")

        plain_tot = 0
        plain_same = 0
        for i in range(len(self.plain_msg_list)):
            tmp_msg = self.plain_msg_list[i]
            for ele in dup_dict:
                if (int(tmp_msg[ele[0]]) + int(tmp_msg[ele[1]])) % 2 == 0:
                    plain_same += 1
                plain_tot += 1
        print(f"plain_tot={plain_tot} plain_same={plain_same}")

        v15_tot = 0
        v15_same = 0
        for i in range(len(self.inv_msg_list_v1_5)):
            tmp_msg = self.inv_msg_list_v1_5[i]
            for ele in dup_dict:
                if (int(tmp_msg[ele[0]]) + int(tmp_msg[ele[1]])) % 2 == 0:
                    v15_same += 1
                v15_tot += 1
        print(f"v15_tot={v15_tot} v15_same={v15_same}")

        v2_tot = 0
        v2_same = 0
        for i in range(len(self.inv_msg_list_v2)):
            tmp_msg = self.inv_msg_list_v2[i]
            for ele in dup_dict:
                if (int(tmp_msg[ele[0]]) + int(tmp_msg[ele[1]])) % 2 == 0:
                    v2_same += 1
                v2_tot += 1
        print(f"v2_tot={v2_tot} v2_same={v2_same}")




        return msg_tot, msg_same, inv_tot, inv_same, plain_tot, plain_same, v15_tot, v15_same, v2_tot, v2_same, len(dup_dict)


    def avg_err_num(self):
        tot = 0
        for i in range(len(self.origin_msg_list)):
            sum = 0
            for j in range(len(self.origin_msg_list[0])):
                if (self.origin_msg_list[i][j] + self.otp[j] + self.inv_msg_list[i][j]) % 2 == 1:
                    sum += 1
            tot += sum
        tot /= len(self.origin_msg_list)
        return tot

    def avg_err_num_v1_5(self):
        tot = 0
        for i in range(len(self.origin_msg_list)):
            sum = 0
            for j in range(len(self.origin_msg_list[0])):
                if (self.origin_msg_list[i][j] + self.otp[j] + self.inv_msg_list_v1_5[i][j]) % 2 == 1:
                    sum += 1
            tot += sum
        tot /= len(self.origin_msg_list)
        return tot

    def avg_err_num_v2(self):
        tot = 0
        for i in range(len(self.origin_msg_list)):
            sum = 0
            for j in range(len(self.origin_msg_list[0])):
                if (self.origin_msg_list[i][j] + self.otp[j] + self.inv_msg_list_v2[i][j]) % 2 == 1:
                    sum += 1
            tot += sum
        tot /= len(self.origin_msg_list)
        return tot

    def msg_recovery(self):
        pass

