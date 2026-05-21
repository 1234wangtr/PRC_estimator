
import re
import pandas as pd
res = []
for i in range(100):
    file = f"gim/data/for_attack3_prc_num_1_steps_50_fpr_1e-05_nowm_0/inv_lat_0.0941/{i}.txt"
    try:
        with open(file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file}")
        continue
    for line in lines:
        '''
        Step 1/100, Loss: 3.3996517658233643, Error: 0.9102,Actual Error: 0.9141, PSNR: 44.6145, Detection: True; Decoding: True; Detection_hard: True; Decoding_hard: True
        '''
        
        pattern = r"Step (\d+)/100, Loss: ([\d\.]+), Error: ([\d\.]+),Actual Error: ([\d\.]+), PSNR: ([\d\.]+), Detection: (True|False); Decoding: (True|False); Detection_hard: (True|False); Decoding_hard: (True|False)"
        match = re.match(pattern, line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            error = float(match.group(3))
            actual_error = float(match.group(4))
            psnr = float(match.group(5))
            detection = match.group(6) == "True"
            decoding = match.group(7) == "True"
            detection_hard = match.group(8) == "True"
            decoding_hard = match.group(9) == "True"
            res.append({
                "image": i,
                "step": step,
                "loss": loss,
                "error": error,
                "actual_error": actual_error,
                "psnr": psnr,
                "detection": detection,
                "decoding": decoding,
                "detection_hard": detection_hard,
                "decoding_hard": decoding_hard
            })
        else:
            '''
            Step 30/30, Loss: 1.2975122928619385, Target Error: 0.3242, PSNR: 30.0688, Detection: True; Decoding: True; Detection_hard: True; Decoding_hard: False
            '''
            pattern = r"Step (\d+)/30, Loss: ([\d\.]+), Target Error: ([\d\.]+), PSNR: ([\d\.]+), Detection: (True|False); Decoding: (True|False); Detection_hard: (True|False); Decoding_hard: (True|False)"
            match = re.match(pattern, line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                error = float(match.group(3))
                actual_error = float(match.group(3))
                psnr = float(match.group(4))
                detection = match.group(5) == "True"
                decoding = match.group(6) == "True"
                detection_hard = match.group(7) == "True"
                decoding_hard = match.group(8) == "True"
                res.append({
                    "image": i,
                    "step": step,
                    "loss": loss,
                    "error": error,
                    "actual_error": actual_error,
                    "psnr": psnr,
                    "detection": detection,
                    "decoding": decoding,
                    "detection_hard": detection_hard,
                    "decoding_hard": decoding_hard
                })
            else:
                print(f"Line does not match pattern: {line}")
    res.append({
        "image": i,
        "step": 999,
        "loss": 999,
        "error": 999,
        "actual_error": 999,
        "psnr": 999,
        "detection": False,
        "decoding": False,
        "detection_hard": False,
        "decoding_hard": False
    })
df = pd.DataFrame(res)

df_false = df[df["detection"] == False]
df_agg = df_false.groupby("image").agg({"step": "first", "error": "first", "actual_error": "first", "psnr": "first", "detection": "first", "decoding": "first", "detection_hard": "first", "decoding_hard": "first"}).reset_index()
succ = df_agg["step"] != 999
df_agg_success = df_agg[succ]
print(succ.mean())
print(df_agg_success[["step", "error", "actual_error", "psnr"]].describe())

