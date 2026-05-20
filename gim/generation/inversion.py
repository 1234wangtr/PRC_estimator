import torch
from diffusers import DPMSolverMultistepScheduler

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from optim_utils import set_random_seed, transform_img, get_dataset


def stable_diffusion_pipe(
        solver_order=1,
        model_id='stabilityai/stable-diffusion-2-1-base',
        cache_dir='',
):
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler(
        beta_end=0.012,
        beta_schedule='scaled_linear',
        beta_start=0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        trained_betas=None,
        solver_order=solver_order,
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    return pipe


def generate(
        image_num=0,
        prompt=None,
        guidance_scale=3.0,
        num_inference_steps=50,
        solver_order=1,
        image_length=512,
        datasets='Gustavosta/Stable-Diffusion-Prompts',
        model_id='stabilityai/stable-diffusion-2-1-base',
        gen_seed=0,
        pipe=None,
        init_latents=None,
):
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pipe is None:
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            trained_betas=None,
            solver_order=solver_order,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )
    pipe = pipe.to(device)

    # load dataset and prompt
    if prompt is None:
        dataset, prompt_key = get_dataset(datasets)
        prompt = dataset[image_num][prompt_key]

    # generate init latent
    seed = gen_seed + image_num
    set_random_seed(seed)

    if init_latents is None:
        init_latents = pipe.get_random_latents()

    # generate image
    output, _ = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=image_length,
        width=image_length,
        latents=init_latents.to(pipe.unet.dtype),
    )
    image = output.images[0]

    return image, prompt, init_latents


def exact_inversion(
        image=None,
        image_tensor=None,
        prompt='',
        guidance_scale=3.0,
        num_inference_steps=50,
        solver_order=1,
        test_num_inference_steps=50,
        inv_order=1,
        decoder_inv=True,
        model_id='stabilityai/stable-diffusion-2-1-base',
        pipe=None,
):
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pipe is None:
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            trained_betas=None,
            solver_order=solver_order,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )
    pipe = pipe.to(device)
    text_embeddings_tuple = pipe.encode_prompt(
        prompt, 'cuda', 1, guidance_scale > 1.0, None
    )
    text_embeddings = torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])

    # image to latent
    image = transform_img(image).unsqueeze(0) if image_tensor is None else image_tensor
    image =  image.to(torch.bfloat16).to(device)
    if decoder_inv:
        image_latents = pipe.decoder_inv(image)
    else:
        image_latents = pipe.get_image_latents(image, sample=False)
    # forward diffusion : image to noise
    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        #text_embeds=text_embeddings,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=test_num_inference_steps,
        inverse_opt=(inv_order != 0),
        inv_order=inv_order
    )

    return reversed_latents
def exact_inversion_with_grad(
        image=None,
        image_tensor=None,
        prompt='',
        guidance_scale=3.0,
        num_inference_steps=50,
        solver_order=1,
        test_num_inference_steps=50,
        inv_order=1,
        decoder_inv=True,
        model_id='stabilityai/stable-diffusion-2-1-base',
        pipe=None,
):
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pipe is None:
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            trained_betas=None,
            solver_order=solver_order,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )
    pipe = pipe.to(device)
    text_embeddings_tuple = pipe.encode_prompt(
        prompt, 'cuda', 1, guidance_scale > 1.0, None
    )
    text_embeddings = torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])

    # image to latent
    image = transform_img(image).unsqueeze(0) if image_tensor is None else image_tensor
    image =  image.to(text_embeddings.dtype).to(device)
    if decoder_inv:
        image_latents = pipe.decoder_inv(image)
    else:
        image_latents = pipe.get_image_latents_with_grad(image, sample=False)
    # forward diffusion : image to noise
    reversed_latents = pipe.forward_diffusion_with_grad(
        latents=image_latents,
        #text_embeds=text_embeddings,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=test_num_inference_steps,
        inverse_opt=(inv_order != 0),
        inv_order=inv_order
    )

    return reversed_latents