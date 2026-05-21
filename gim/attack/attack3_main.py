import sys
sys.path.append("gim/generation")
import os
import argparse
import pickle
import torch
from PIL import Image
from tqdm import tqdm
from prc import Detect, Decode
import pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, exact_inversion, generate

parser = argparse.ArgumentParser("Args")
parser.add_argument("--test_num", type=int, default=1)
parser.add_argument("--method", type=str, default="prc")  
parser.add_argument(
    "--model_id",
    type=str,
    default="stabilityai/stable-diffusion-2-1-base",
)
parser.add_argument(
    "--dataset_id", type=str, default="Gustavosta/Stable-Diffusion-Prompts"
)
parser.add_argument("--inf_steps", type=int, default=50)
parser.add_argument("--nowm", type=int, default=0)
parser.add_argument("--fpr", type=str, default=0.00001)

parser.add_argument("--prc_t", type=int, default=3)
parser.add_argument("--eps", type=float, default=0.0941)

parser.add_argument("--test_path", type=str, default="original_images")
args, unknown = parser.parse_known_args()

try:
    args.fpr = float(args.fpr)
except ValueError:
    args.fpr = 0.00001
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
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

with open(f"{save_folder}/keys/{exp_id}.pkl", "rb") as f:
    encoding_key, decoding_key = pickle.load(f)

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id)
pipe.set_progress_bar_config(disable=True)

cur_inv_order = 0
var = 1.5
combined_results = []
import json


eps = args.eps  
os.makedirs(f"{save_folder}/inv_lat_{eps}", exist_ok=True)
os.makedirs(f"{save_folder}/adv_img_{eps}", exist_ok=True)
with open(f"{save_folder}/prompts.json", "r") as f:
    prompts = json.load(f)


with torch.no_grad():
    for i in tqdm(range(test_num)):
        gen_img = Image.open(f"{save_folder}/gen_img/{i}.png")
        gen_lat = torch.load(f"{save_folder}/gen_lat/{i}.pt")
        x = torch.load(f"{save_folder}/gen_x/{i}.pt")
        x_otp = torch.load(f"{save_folder}/gen_x_otp/{i}.pt")
        pipe.unet.to(torch.bfloat16)
        pipe.vae.to(torch.bfloat16)
        reversed_latents = exact_inversion(
            gen_img,
            prompt="",
            test_num_inference_steps=args.inf_steps,
            inv_order=cur_inv_order,
            pipe=pipe,
        )
        from torchvision import transforms

        target = (
            (x_otp.flatten() - 0.5).sign().reshape(reversed_latents.shape).to(device)
        )

        def tanh_approximation_loss(x, y, k=5.0):
            return torch.mean((torch.tanh(k * x) - y) ** 2)

        loss = tanh_approximation_loss(reversed_latents, target)

        image_tensor_original = (
            transforms.ToTensor()(gen_img).unsqueeze(0).bfloat16().to(device)
        )
        image_tensor_original = image_tensor_original * 2 - 1  
        image_tensor = image_tensor_original.clone().detach()
        steps = 100
        lr = 0.01
        import numpy as np

        def psnr(img1, img2):
            mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
            return 20 * np.log10(255.0 / np.sqrt(mse))

        from galois import GF
        from pathlib import Path
        from inversion import exact_inversion_with_grad

        os.remove(f"{save_folder}/inv_lat_{eps}/{i}.txt") if Path(
            f"{save_folder}/inv_lat_{eps}/{i}.txt"
        ).exists() else None
        with torch.set_grad_enabled(True):
            for step in range(steps):
                image_tensor = image_tensor.to(device).requires_grad_(True)
                new_latent = exact_inversion_with_grad(
                    image_tensor=image_tensor,
                    prompt="",
                    test_num_inference_steps=50,
                    inv_order=cur_inv_order,
                    pipe=pipe,
                    decoder_inv=False,
                )
                loss = tanh_approximation_loss(new_latent, target)
                loss.backward()
                with torch.no_grad():
                    
                    
                    grad_sign = image_tensor.grad.sign()
                    image_tensor.grad.zero_()
                    
                    image_tensor -= lr * grad_sign

                    eta = torch.clamp(image_tensor - image_tensor_original, -eps, eps)
                    image_tensor = torch.clamp(
                        image_tensor_original + eta, -1, 1
                    )  
                    
                    new_latent_sign = new_latent.sign()

                    
                    target_error_rate = (
                        (new_latent_sign != target).bfloat16().mean().item()
                    )

                    image = transforms.ToPILImage()(
                        image_tensor.squeeze().cpu() * 0.5 + 0.5
                    )
                    psnr_ = psnr(image, gen_img)

                    image.save(f"{save_folder}/adv_img_{eps}/{i}_step_{step + 1}.png")

                    adv_image = Image.open(
                        f"{save_folder}/adv_img_{eps}/{i}_step_{step + 1}.png"
                    )
                    pipe.unet.to(torch.bfloat16)
                    pipe.vae.to(torch.bfloat16)
                    reversed_latents_attacked = exact_inversion(
                        adv_image,
                        prompt="",
                        test_num_inference_steps=args.inf_steps,
                        inv_order=cur_inv_order,
                        pipe=pipe,
                    )
                    reversed_prc_attacked = (
                        prc_gaussians.recover_posteriors(
                            reversed_latents_attacked.to(torch.float64).flatten().cpu(),
                            variances=float(var),
                        )
                        .flatten()
                        .cpu()
                    )
                    p, detection_result_attacked = Detect(
                        decoding_key, reversed_prc_attacked, false_positive_rate=fpr
                    )
                    decoding_result_attacked = (
                        Decode(
                            decoding_key, reversed_prc_attacked, print_progress=False
                        )
                        is not None
                    )
                    p_hard, detection_result_attacked_hard = Detect(
                        decoding_key,
                        reversed_prc_attacked.sign(),
                        false_positive_rate=fpr,
                    )
                    decoding_result_attacked_hard = (
                        Decode(
                            decoding_key,
                            reversed_prc_attacked.sign(),
                            print_progress=False,
                        )
                        is not None
                    )
                    actual_error_rate = (
                        (reversed_latents_attacked.sign().flatten() != target.flatten())
                        .bfloat16()
                        .mean()
                        .item()
                    )
                    with open(f"{save_folder}/inv_lat_{eps}/{i}.txt", "a") as f:
                        f.write(
                            f"Step {step + 1}/{steps}, Loss: {loss.item()}, Error: {target_error_rate:.4f},Actual Error: {actual_error_rate:.4f}, PSNR: {psnr_:.4f}, Detection: {detection_result_attacked}; Decoding: {decoding_result_attacked}; Detection_hard: {detection_result_attacked_hard}; Decoding_hard: {decoding_result_attacked_hard}\n"
                        )
                    print(
                        f"Step {step + 1}/{steps}, Loss: {loss.item()}, Target Error: {target_error_rate:.4f}, Actual Error: {actual_error_rate:.4f}, PSNR: {psnr_:.4f}, Detection: {detection_result_attacked}; Decoding: {decoding_result_attacked}; Detection_hard: {detection_result_attacked_hard}; Decoding_hard: {decoding_result_attacked_hard}"
                    )
                    if (not detection_result_attacked) and (
                        not decoding_result_attacked
                    ):
                        break



