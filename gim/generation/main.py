
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
import pickle
import json
from tqdm import tqdm
import random
import numpy as np

from datasets import load_dataset
from prc import KeyGen, Encode, Detect
import pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, generate, exact_inversion

from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=16)
parser.add_argument('--method', type=str, default='prc')
parser.add_argument('--gen_model_id', type=str,
                    default='SD21')
parser.add_argument('--inv_model_ids', type=str,
                    default='SD21')
parser.add_argument('--dataset_id', type=str,
                    default='Gustavosta/Stable-Diffusion-Prompts')  
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=8)
args = parser.parse_args()
print(args)


hf_cache_dir = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
n = 4 * 64 * 64  
method = args.method
test_num = args.test_num
model_id = args.gen_model_id
inv_model_ids = [mid.strip() for mid in args.inv_model_ids.split(',')]
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'
output_dir = f"gim/data/gen_data_{model_id}_t{prc_t}_{exp_id}"
os.makedirs(output_dir, exist_ok=True)
def save_keys_to_json(public_key: np.ndarray, secret_key: csr_matrix, otp: np.ndarray,
                      errors:list,prc_codewords: list, reverse_codes: list, detections: list,
                      reverse_codes_v2: list, detections_v2: list,
                      reverse_codes_v15: list, detections_v15: list,
                      test_num: int, filename="0001.json"):
    public_key_list = public_key.tolist()
    secret_key_list = []
    for i in range(secret_key.shape[0]):
        row_start = secret_key.indptr[i]
        row_end = secret_key.indptr[i + 1]
        indices = secret_key.indices[row_start:row_end]
        sorted_indices = sorted(indices.tolist())
        
        
        
        
        secret_key_list.append(sorted_indices)
    errors_save = [np.where(error==1)[0] for error in errors][0].tolist()
    otp_list = otp.tolist()

    data = {
        "secret_key": secret_key_list,
        "public_key": public_key_list,
        "one_time_pad": otp_list,
        "msg": {
            "num": str(test_num),
            "succ2.1": str(detections.count(True)),
            "succ2": str(detections_v2.count(True)),
            "succ1.5": str(detections_v15.count(True))
        },
        "error": errors_save,
        "prc_codeword_save": prc_codewords,
        "PRC_reverse_code": reverse_codes,
        "detection_result": detections,
        "PRC_reverse_code_v2": reverse_codes_v2,
        "detection_result_v2": detections_v2,
        "PRC_reverse_code_v1_5": reverse_codes_v15,
        "detection_result_v1_5": detections_v15
    }
    
    with open(f"{output_dir}/"+filename, "w") as f:
        json.dump(data, f)



if dataset_id == 'coco':
    with open('coco/captions_val2017.json') as f:
        all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
else:
    all_prompts = [sample['Prompt']
                   for sample in load_dataset(dataset_id)['test']]

prompts = random.sample(all_prompts, test_num)
if "SD21" in inv_model_ids or model_id == "SD21":
    pipe_v2_1 = stable_diffusion_pipe(
        solver_order=1, model_id="/data/huggingface-mirror/dataroot/models/stabilityai/stable-diffusion-2-1-base/", cache_dir=hf_cache_dir)
    pipe_v2_1.set_progress_bar_config(disable=True)
else:
    pipe_v2_1 = None
if "SD15" in inv_model_ids or model_id == "SD15":
    pipe_v1_5 = stable_diffusion_pipe(solver_order=1,
                                         model_id="/data/huggingface-mirror/dataroot/models/sd-legacy/stable-diffusion-v1-5/",
                                      cache_dir=hf_cache_dir)
    pipe_v1_5.set_progress_bar_config(disable=True)
else:
    pipe_v1_5 = None

if "SD2" in inv_model_ids or model_id == "SD2":
    pipe_v2 = stable_diffusion_pipe(solver_order=1,
                                    model_id="/data/huggingface-mirror/dataroot/models/Manojb/stable-diffusion-2-base/",
                                    cache_dir=hf_cache_dir)
    pipe_v2.set_progress_bar_config(disable=True)
else:
    pipe_v2 = None
if model_id == "SD15":
    pipe_gen = pipe_v1_5
elif model_id == "SD21":
    pipe_gen = pipe_v2_1
elif model_id == "SD2":
    pipe_gen = pipe_v2
else:
    raise NotImplementedError


def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed


var = 1.5
cur_inv_order = 0

for i in range(args.start, args.end):
    seed_everything(42+1000*i)
    (encoding_key_ori, decoding_key_ori) = KeyGen(
        n, false_positive_rate=fpr, t=prc_t)
    encoding_key = encoding_key_ori
    decoding_key = decoding_key_ori
    encoding_key_save = encoding_key_ori[0]
    decoding_key_save = decoding_key_ori[1]
    OTP = encoding_key_ori[1]
    prc_codewords = []
    errors = []
    reverse_codes_2_1 = []
    reverse_codes_2 = []
    reverse_codes_1_5 = []

    detection_results_2_1 = []
    detection_results_2 = []
    detection_results_1_5 = []
    filename = f"{i:04d}.json"
    for j in tqdm(range(test_num)):
        current_prompt = prompts[j]
        if nowm:
            seed_everything(42+1000*j)
            init_latents_np = np.random.randn(1, 4, 64, 64)
            init_latents = torch.from_numpy(
                init_latents_np).to(torch.float64).to(device)
        else:
            if method == 'prc':
                prc_codeword_save, prc_codeword, error,payload = Encode(
                    encoding_key) 
                errors.append(error.tolist())
                prc_codewords.append((0.5-prc_codeword_save/2).int().tolist())
                init_latents = prc_gaussians.sample(
                    prc_codeword_save).reshape(1, 4, 64, 64).to(device)
            else:
                raise NotImplementedError
        
        
        orig_image, _, _ = generate(prompt=current_prompt,
                                    init_latents=init_latents,
                                    num_inference_steps=args.inf_steps,
                                    solver_order=1,
                                    pipe=pipe_gen
                                    )
        
        if "SD21" in inv_model_ids:
            reversed_latents = exact_inversion(
                orig_image,
                prompt='',
                test_num_inference_steps=args.inf_steps,
                inv_order=cur_inv_order,
                pipe=pipe_v2_1
            )
            reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(
                torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
            PRC_reverse_code, detection_result = Detect(decoding_key, reversed_prc)
            reverse_codes_2_1.append(PRC_reverse_code)
            detection_results_2_1.append(bool(detection_result))
        if "SD2" in inv_model_ids:
            reversed_latents = exact_inversion(
                orig_image,
                prompt='',
                test_num_inference_steps=args.inf_steps,
                inv_order=cur_inv_order,
                pipe=pipe_v2
            )
            reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(
                torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
            PRC_reverse_code, detection_result = Detect(decoding_key, reversed_prc)
            reverse_codes_2.append(PRC_reverse_code)
            detection_results_2.append(bool(detection_result))

        if "SD15" in inv_model_ids:
            reversed_latents = exact_inversion(
                orig_image,
                prompt='',
                test_num_inference_steps=args.inf_steps,
                inv_order=cur_inv_order,
                pipe=pipe_v1_5
            )
            reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(
                torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
            PRC_reverse_code, detection_result = Detect(decoding_key, reversed_prc)
            reverse_codes_1_5.append(PRC_reverse_code)
            detection_results_1_5.append(bool(detection_result))

        
        save_keys_to_json(encoding_key_save, decoding_key_save, OTP,errors,
                        prc_codewords, reverse_codes_2_1, detection_results_2_1, reverse_codes_2, detection_results_2,
                        reverse_codes_1_5, detection_results_1_5,
                        test_num, filename=filename)
    print(f"Saved key and results to {filename}")
