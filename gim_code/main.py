import os
from scipy.sparse import csr_matrix
from inversion import stable_diffusion_pipe, exact_inversion
from src.prc import Detect, Decode
from inversion import stable_diffusion_pipe, generate
import src.pseudogaussians as prc_gaussians
from src.prc import KeyGen, Encode, str_to_bin, bin_to_str
from datasets import load_dataset
import numpy as np
import random
from tqdm import tqdm
import json
import pickle
import torch
import argparse
# original_fn = np.random.seed
# def hook_seed(seed):
#     print("set seed:", seed)
#     original_fn(seed)
# np.random.seed = hook_seed
# hook_seed(???)


parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=256)
parser.add_argument('--method', type=str, default='prc')  # gs, tr, prc
parser.add_argument('--model_id', type=str,
                    default='stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str,
                    default='Gustavosta/Stable-Diffusion-Prompts')  # coco
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
args = parser.parse_args()
print(args)
start = 400
num = 5

hf_cache_dir = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'

# def compare_arrays(float_arr, int_arr):
#     if len(float_arr) != len(int_arr):
#         raise ValueError("Arrays must have the same length")

#     if not float_arr or not int_arr:
#         return 0.0

#     total_positions = len(float_arr)
#     matching_positions = sum(1 for f, i in zip(float_arr, int_arr) if int(f) == i)
#     error_count = total_positions - matching_positions
#     error_rate = error_count / total_positions

#     return error_rate



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
        # if len(sorted_indices) >= 3:
        #     group = [sorted_indices[2], sorted_indices[0], sorted_indices[1]]
        # else:
        #     group = sorted_indices + [0] * (3 - len(sorted_indices))
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
    os.makedirs("keys-t3-256", exist_ok=True)
    with open("keys-t3-256/"+filename, "w") as f:
        json.dump(data, f, indent=2)
    os.makedirs("paperfig-t3-256", exist_ok=True)
    with open("paperfig-t3-256/"+filename, "w") as f:
        json.dump(data, f, indent=2)


if dataset_id == 'coco':
    save_folder = f'./results-t3-256/{exp_id}_coco/original_images'
else:
    save_folder = f'./results-t3-256/{exp_id}/original_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(f'Saving original images to {save_folder}')


if dataset_id == 'coco':
    with open('coco/captions_val2017.json') as f:
        all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
else:
    all_prompts = [sample['Prompt']
                   for sample in load_dataset(dataset_id)['test']]

prompts = random.sample(all_prompts, test_num)

pipe = stable_diffusion_pipe(
    solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
# pipe_v1_5 = stable_diffusion_pipe(solver_order=1,
#                                   model_id=model_id,
#                                   #    model_id="sd-legacy/stable-diffusion-v1-5",
#                                   cache_dir=hf_cache_dir)
# pipe_v2 = stable_diffusion_pipe(solver_order=1,
#                                 model_id=model_id,
#                                 # model_id="stabilityai/stable-diffusion-2-base",
#                                 cache_dir=hf_cache_dir)
# pipi_xl_1 = stable_diffusion_pipe(solver_order=1, model_id="stabilityai/stable-diffusion-xl-base-1.0", cache_dir=hf_cache_dir)

pipe.set_progress_bar_config(disable=True)
# pipe_v2.set_progress_bar_config(disable=True)
# pipi_xl_1.set_progress_bar_config(disable=True)

# pipe_v1_5.set_progress_bar_config(disable=True)


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

for i in range(start, start+num):
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
                prc_codewords.append(prc_codeword_save.tolist())
                init_latents = prc_gaussians.sample(
                    prc_codeword).reshape(1, 4, 64, 64).to(device)
            else:
                raise NotImplementedError
        current_prompt = "the day that aliens met us, digital art, epic, lighting, color harmony, volumetric lighting, matte painting, chromatic aberration, shallow depth of field, epic composition, trending on artstation"
        orig_image, _, _ = generate(prompt=current_prompt,
                                    init_latents=init_latents,
                                    num_inference_steps=args.inf_steps,
                                    solver_order=1,
                                    pipe=pipe
                                    )
        orig_image.save(f'paperfig/nowatermark/{j}.png')

        reversed_latents = exact_inversion(
            orig_image,
            prompt='',
            test_num_inference_steps=args.inf_steps,
            inv_order=cur_inv_order,
            pipe=pipe
        )
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(
            torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
        PRC_reverse_code, detection_result = Detect(decoding_key, reversed_prc)
        reverse_codes_2_1.append(PRC_reverse_code)
        detection_results_2_1.append(bool(detection_result))

        # reversed_latents = exact_inversion(
        #     orig_image,
        #     prompt='',
        #     test_num_inference_steps=args.inf_steps,
        #     inv_order=cur_inv_order,
        #     pipe=pipe_v2
        # )
        # reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(
        #     torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
        # PRC_reverse_code, detection_result = Detect(decoding_key, reversed_prc)
        # reverse_codes_2.append(PRC_reverse_code)
        # detection_results_2.append(bool(detection_result))

        # reversed_latents = exact_inversion(
        #     orig_image,
        #     prompt='',
        #     test_num_inference_steps=args.inf_steps,
        #     inv_order=cur_inv_order,
        #     pipe=pipe_v1_5
        # )
        # reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(
        #     torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
        # PRC_reverse_code, detection_result = Detect(decoding_key, reversed_prc)
        # reverse_codes_1_5.append(PRC_reverse_code)
        # detection_results_1_5.append(bool(detection_result))

    filename = f"{i+2:04d}.json"
    save_keys_to_json(encoding_key_save, decoding_key_save, OTP,errors,
                      prc_codewords, reverse_codes_2_1, detection_results_2_1, reverse_codes_2, detection_results_2,
                      reverse_codes_1_5, detection_results_1_5,
                      test_num, filename=filename)
    print(f"Saved key and results to {filename}")
