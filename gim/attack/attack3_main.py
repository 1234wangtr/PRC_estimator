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
parser.add_argument("work_dir", type=str, help="Directory containing files")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10)
parser.add_argument("--method", type=str, default="prc")  
parser.add_argument('--model_id', type=str,
                    default='SD21')
parser.add_argument("--inf_steps", type=int, default=50)
parser.add_argument("--eps", type=float, default=24)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--test_path", type=str, default="original_images")

args, unknown = parser.parse_known_args()
print(args)

steps = args.steps
lr = args.lr
device = "cuda" if torch.cuda.is_available() else "cpu"
n = 4 * 64 * 64  
method = args.method
model_id = args.model_id
work_dir = args.work_dir
with open(f"{work_dir}/keys.pkl", "rb") as f:
    _, decoding_key = pickle.load(f)
fpr = decoding_key[3]
print(f"Loaded keys from {work_dir}/keys.pkl with false positive rate {fpr}")
if model_id == 'SD21':
    pipe = stable_diffusion_pipe(solver_order=1, model_id='.models/Manojb/stable-diffusion-2-1-base/')
else:
    raise NotImplementedError
pipe.set_progress_bar_config(disable=True)

cur_inv_order = 0
var = 1.5
combined_results = []
import json


eps = args.eps  
os.makedirs(f"{work_dir}/inv_lat_{eps}", exist_ok=True)
os.makedirs(f"{work_dir}/adv_img_{eps}", exist_ok=True)

start = args.start
end = args.end

with torch.no_grad():
    for i in tqdm(range(start, end)):
        gen_img = Image.open(f"{work_dir}/gen_img/{i}.png")
        gen_lat = torch.load(f"{work_dir}/gen_lat/{i}.pt")
        x = torch.load(f"{work_dir}/gen_x/{i}.pt")
        x_otp = torch.load(f"{work_dir}/gen_x_otp/{i}.pt")
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
        
        import numpy as np

        def psnr(img1, img2):
            mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
            return 20 * np.log10(255.0 / np.sqrt(mse))

        from galois import GF
        from pathlib import Path
        from inversion import exact_inversion_with_grad

        os.remove(f"{work_dir}/inv_lat_{eps}/{i}.txt") if Path(
            f"{work_dir}/inv_lat_{eps}/{i}.txt"
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

                    image.save(f"{work_dir}/adv_img_{eps}/{i}_step_{step + 1}.png")

                    Path(f"{work_dir}/adv_img_{eps}/{i}_step_{step}.png").unlink(missing_ok=True)
                    
                    adv_image = Image.open(
                        f"{work_dir}/adv_img_{eps}/{i}_step_{step + 1}.png"
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
                    with open(f"{work_dir}/inv_lat_{eps}/{i}.txt", "a") as f:
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


