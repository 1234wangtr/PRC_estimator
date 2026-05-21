
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
from prc import KeyGen, Encode
import pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, generate


parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=1)
parser.add_argument('--method', type=str, default='prc') 
parser.add_argument('--model_id', type=str, default='/data/huggingface-mirror/dataroot/models/stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') 
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n = 4 * 64 * 64  
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'


save_folder = f'gim/data/for_attack3_{exp_id}'
os.makedirs(save_folder+"/keys", exist_ok=True)
os.makedirs(save_folder+"/gen_img", exist_ok=True)
os.makedirs(save_folder+"/gen_lat", exist_ok=True)
os.makedirs(save_folder+"/gen_x", exist_ok=True)
os.makedirs(save_folder+"/gen_x_otp", exist_ok=True)

if method == 'prc':
    if not os.path.exists(f'{save_folder}/keys/{exp_id}.pkl'):  
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, false_positive_rate=fpr, t=prc_t)  
        with open(f'{save_folder}/keys/{exp_id}.pkl', 'wb') as f:  
            pickle.dump((encoding_key_ori, decoding_key_ori), f)
        with open(f'{save_folder}/keys/{exp_id}.pkl', 'rb') as f:  
            encoding_key, decoding_key = pickle.load(f)
            print("encoding_key",encoding_key)
            print("decoding_key",decoding_key)
        assert encoding_key[0].all() == encoding_key_ori[0].all()
    else:  
        with open(f'{save_folder}/keys/{exp_id}.pkl', 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
            print("encoding_key",encoding_key)
            print("decoding_key",decoding_key)
        print(f'Loaded PRC keys from file {save_folder}/keys/{exp_id}.pkl')
else:
    raise NotImplementedError


print(f'Saving original images to {save_folder}')

random.seed(42)

all_prompts = [sample['Prompt'] for sample in load_dataset(dataset_id)['test']]

prompts = random.sample(all_prompts, test_num)
import json
with open(f'{save_folder}/prompts.json', 'w') as f:
    json.dump(prompts, f, indent=4)
pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id)
pipe.set_progress_bar_config(disable=True)

def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed

for i in tqdm(range(test_num)):
    seed_everything(i)
    current_prompt = prompts[i]
    if nowm:
        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float64).to(device)
    else:
        if method == 'prc':
            prc_codeword,x,otp,error = Encode(encoding_key)
            print("prc_codeword",prc_codeword)
            
            init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64).to(device).to(torch.bfloat16)
        else:
            raise NotImplementedError
    orig_image, _, _ = generate(prompt=current_prompt,
                                init_latents=init_latents,
                                num_inference_steps=args.inf_steps,
                                solver_order=1,
                                pipe=pipe
                                )
    orig_image.save(f'{save_folder}/gen_img/{i}.png')
    torch.save(torch.tensor(x).cpu(), f'{save_folder}/gen_x/{i}.pt')
    torch.save(torch.tensor(x+otp).cpu(), f'{save_folder}/gen_x_otp/{i}.pt')
    torch.save(init_latents.cpu(), f'{save_folder}/gen_lat/{i}.pt')

print(f'Done generating {method} images')