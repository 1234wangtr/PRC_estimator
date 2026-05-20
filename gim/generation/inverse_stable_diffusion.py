from typing import Callable, List, Optional, Union, Tuple
import copy
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modified_stable_diffusion import ModifiedStableDiffusionPipeline

from contextlib import nullcontext

### credit to: https://github.com/cccntu/efficient-prompt-to-prompt
    
class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super(InversableStableDiffusionPipeline, self).__init__(vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                requires_safety_checker)

    def get_random_latents(self, latents=None, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1
        device = self._execution_device

        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        return self.get_image_latents_with_grad(image, sample, rng_generator)
    def get_image_latents_with_grad(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents
    
    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        self.vae = self.vae.bfloat16()
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    def decode_image_for_gradient_float(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        vae = copy.deepcopy(self.vae).bfloat16()
        image = [
            vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    def apply_guidance_scale(self, model_output, guidance_scale):
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise_pred
        else:
            return model_output         
    
    @torch.inference_mode() 
    def forward_diffusion(
        self,
        use_old_emb_i=25,  
        text_embeddings=None,  # 当前使用的文本嵌入（条件信息）
        old_text_embeddings=None,  
        new_text_embeddings=None,  
        latents: Optional[torch.FloatTensor] = None,  # 初始的潜变量（噪声）输入
        num_inference_steps: int = 10,  # 反演时所用的扩散步数
        guidance_scale: float = 7.5,  # 指导尺度参数，用于 classifier-free guidance
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,  # 可选的回调函数，用于监控反演过程
        callback_steps: Optional[int] = 1,  # 回调函数的调用间隔步数
        inverse_opt=True,  # 是否启用反演优化（固定点校正）
        inv_order=None,  # 反演方法的阶数，若未指定则使用调度器默认值
        **kwargs,
    ):
        return self.forward_diffusion_with_grad(
            use_old_emb_i=use_old_emb_i,
            text_embeddings=text_embeddings,
            old_text_embeddings=old_text_embeddings,
            new_text_embeddings=new_text_embeddings,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            inverse_opt=inverse_opt,
            inv_order=inv_order,
            **kwargs,
        )
    def forward_diffusion_with_grad(
        self,
        use_old_emb_i=25,  
        text_embeddings=None,  # 当前使用的文本嵌入（条件信息）
        old_text_embeddings=None,  
        new_text_embeddings=None,  
        latents: Optional[torch.FloatTensor] = None,  # 初始的潜变量（噪声）输入
        num_inference_steps: int = 10,  # 反演时所用的扩散步数
        guidance_scale: float = 7.5,  # 指导尺度参数，用于 classifier-free guidance
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,  # 可选的回调函数，用于监控反演过程
        callback_steps: Optional[int] = 1,  # 回调函数的调用间隔步数
        inverse_opt=True,  # 是否启用反演优化（固定点校正）
        inv_order=None,  # 反演方法的阶数，若未指定则使用调度器默认值
        **kwargs,
    ):
        with nullcontext():
        # with torch.no_grad():
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            self.scheduler.set_timesteps(num_inference_steps)
            timesteps_tensor = self.scheduler.timesteps.to(self.device)
            sigma = torch.tensor(self.scheduler.init_noise_sigma, device=self.device, dtype=latents.dtype)
            latents_aa = latents * sigma

            # 判断是否存在 prompt-to-prompt 情景：即同时提供了旧的和新的文本嵌入
            if old_text_embeddings is not None and new_text_embeddings is not None:
                prompt_to_prompt = True
            else:
                prompt_to_prompt = False

            if inv_order is None:
                inv_order = self.scheduler.solver_order
    
            
            timesteps_tensor = reversed(timesteps_tensor) # inversion process  # 将时间步倒序，用于反演过程（从最后一步开始逆向更新）

            self.unet = self.unet.bfloat16()
            latents_aa = latents_aa.bfloat16()
            text_embeddings = text_embeddings.bfloat16()
            # 遍历每一个反向时间步，同时显示进度条
            for i, t in enumerate(self.progress_bar(timesteps_tensor)):
                if prompt_to_prompt:
                    if i < use_old_emb_i:
                        text_embeddings = old_text_embeddings
                    else:
                        text_embeddings = new_text_embeddings
                # 根据当前时间步 t 和调度器配置，计算前一时间步 prev_timestep
                prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
                )

                # call the callback, if provided  # 如果提供了回调函数，并且当前步数满足调用间隔，则执行回调函数
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents_aa)
                

                # Our Algorithm
                # 开始反演算法
                # Algorithm 1：低阶反演更新方法，当 inv_order 小于2 或 inv_order 等于2且在第一个步骤时使用
                if inv_order < 2 or (inv_order == 2 and i == 0):
                    s = t  # 将当前时间步保存到 s
                    t = prev_timestep  # 将 t 更新为前一时间步

                    # 从调度器中提取当前时间步 s 和前一时间步 t 对应的参数
                    lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                    sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                    h = lambda_t - lambda_s  # 计算两个时间步的 lambda 差值
                    alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                    phi_1 = torch.expm1(-h)  # 计算修正项 phi_1 = exp(-h) - 1

                    # 如果启用了 classifier-free guidance，则将潜变量复制两份作为输入
                    latent_model_input = torch.cat([latents_aa] * 2) if do_classifier_free_guidance else latents_aa
                    # 对输入进行尺度调整，以匹配当前时间步 t 的要求
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # 调用 UNet 模型预测噪声残差，输入为处理后的潜变量、时间步 s 以及文本嵌入
                    noise_pred = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample
                    # 对预测的噪声进行 guidance 调整
                    noise_pred = self.apply_guidance_scale(noise_pred, guidance_scale)
                    # 将噪声预测转换为模型输出，使用调度器中的转换函数
                    model_s = self.scheduler.convert_model_output(noise_pred, t, latents_aa)
                    x_t = latents_aa  # 保存当前潜变量状态作为 x_t

                    # 利用调度器公式更新潜变量：
                    # latents = (sigma_s / sigma_t) * (latents + alpha_t * phi_1 * model_s)
                    latents_aa = (sigma_s / sigma_t) * (latents_aa + alpha_t * phi_1 * model_s)

                    # 如果启用反演优化，则进行固定点校正，进一步修正潜变量
                    if (inverse_opt):
                        # 如果 inv_order 为 2 且在第一个步骤，则使用较大的步长进行校正
                        if (inv_order == 2 and i == 0):
                            latents_aa = self.fixedpoint_correction(latents_aa, s, t, x_t, order=1, text_embeddings=text_embeddings, guidance_scale=guidance_scale,
                                                                step_size=1, scheduler=True)
                        else:
                            # 否则使用较小步长进行固定点校正
                            latents_aa = self.fixedpoint_correction(latents_aa, s, t, x_t, order=1, text_embeddings=text_embeddings, guidance_scale=guidance_scale,
                                                                step_size=0.5, scheduler=True)


                # Algorithm 2：高阶反演方法，当 inv_order 等于 2 且不在第一步时使用
                elif inv_order == 2:
                    with torch.no_grad():  # 再次确保在无梯度模式下进行
                        # 当当前不是最后一个时间步时（i+1 小于总步数）
                        if (i + 1 < len(timesteps_tensor)):
                            y = latents_aa.clone()  # 克隆当前潜变量作为中间变量 y

                            s = t  # 保存当前时间步到 s
                            t = prev_timestep  # 更新 t 为前一时间步
                            r = timesteps_tensor[i + 1] if i+1 < len(timesteps_tensor) else 0  # 获取下一个时间步 r（若存在，否则为0）

                            # Line 3 ~ 6：细粒度的 naive DDIM 反演更新
                            for tt in range(t, s, 10):  # 每隔 10 个步长进行一次更新，从 t 到 s
                                ss = tt + 10  # 定义当前步长后的步长 ss
                                # 提取对应的调度器参数
                                lambda_s, lambda_t = self.scheduler.lambda_t[ss], self.scheduler.lambda_t[tt]
                                sigma_s, sigma_t = self.scheduler.sigma_t[ss], self.scheduler.sigma_t[tt]
                                h = lambda_t - lambda_s  # 计算步长差 h
                                alpha_s, alpha_t = self.scheduler.alpha_t[ss], self.scheduler.alpha_t[tt]
                                phi_1 = torch.expm1(-h)  # 计算修正项

                                # 如果启用了 guidance，则复制 y 两份
                                y_input = torch.cat([y] * 2) if do_classifier_free_guidance else y
                                # 对 y_input 进行尺度调整以适应当前步长 tt
                                y_input = self.scheduler.scale_model_input(y_input, tt)

                                # 调用 UNet 预测噪声，并应用 guidance
                                noise_pred = self.unet(y_input, ss, encoder_hidden_states=text_embeddings).sample
                                noise_pred = self.apply_guidance_scale(noise_pred, guidance_scale)
                                # 转换噪声为模型输出
                                model_s = self.scheduler.convert_model_output(noise_pred, tt, y)
                                # 更新 y，根据公式： y = (sigma_s / sigma_t) * (y + alpha_t * phi_1 * model_s)
                                y = (sigma_s / sigma_t) * (y + alpha_t * phi_1 * model_s)
                            y_t = y.clone()  # 保存细粒度更新后的中间结果 y_t

                            # 再次从 s 到 r 进行类似更新
                            for tt in range(s, r, 10):
                                ss = tt + 10
                                lambda_s, lambda_t = self.scheduler.lambda_t[ss], self.scheduler.lambda_t[tt]
                                sigma_s, sigma_t = self.scheduler.sigma_t[ss], self.scheduler.sigma_t[tt]
                                h = lambda_t - lambda_s
                                alpha_s, alpha_t = self.scheduler.alpha_t[ss], self.scheduler.alpha_t[tt]
                                phi_1 = torch.expm1(-h)

                                y_input = torch.cat([y] * 2) if do_classifier_free_guidance else y
                                y_input = self.scheduler.scale_model_input(y_input, tt)

                                model_s = self.unet(y_input, ss, encoder_hidden_states=text_embeddings).sample
                                noise_pred = self.apply_guidance_scale(model_s, guidance_scale)
                                model_s = self.scheduler.convert_model_output(noise_pred, tt, y)
                                y = (sigma_s / sigma_t) * (y + alpha_t * phi_1 * model_s)



                            # Line 8 ~ 12：使用后向 Euler 方法进行更新
                            t = prev_timestep  # 将 t 更新为前一时间步
                            s = timesteps_tensor[i]  # s 为当前时间步（未更新前的值）
                            r = timesteps_tensor[i+1]  # r 为下一个时间步

                            # 提取 s 和 t 对应的参数
                            lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                            sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                            h = lambda_t - lambda_s
                            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                            phi_1 = torch.expm1(-h)

                            x_t = latents_aa  # 保存当前潜变量状态为 x_t

                            # 对 y_t 进行尺度调整后预测噪声
                            y_t_model_input = torch.cat([y_t] * 2) if do_classifier_free_guidance else y_t
                            y_t_model_input = self.scheduler.scale_model_input(y_t_model_input, s)
                            noise_pred = self.unet(y_t_model_input, s, encoder_hidden_states=text_embeddings).sample
                            noise_pred = self.apply_guidance_scale(noise_pred, guidance_scale)
                            model_s_output = self.scheduler.convert_model_output(noise_pred, s, y_t)

                            # 对 y 进行相似处理，针对时间步 r
                            y_model_input = torch.cat([y] * 2) if do_classifier_free_guidance else y
                            y_model_input = self.scheduler.scale_model_input(y_model_input, r)
                            noise_pred = self.unet(y_model_input, r, encoder_hidden_states=text_embeddings).sample
                            noise_pred = self.apply_guidance_scale(noise_pred, guidance_scale)
                            model_r_output = self.scheduler.convert_model_output(noise_pred, r, y)

                            latents_aa = y_t.clone()  # 将更新结果设为新的潜变量，复制 y_t

                            # Line 11：如果开启反演优化，则调用固定点校正进行更新（高阶校正，order==2）
                            if inverse_opt:
                                latents_aa = self.fixedpoint_correction(latents_aa, s, t, x_t, order=2, r=r,
                                                                    model_s_output=model_s_output, model_r_output=model_r_output, text_embeddings=text_embeddings, guidance_scale=guidance_scale,
                                                                    step_size=10/t, scheduler=False)
                            
                        # 当当前步骤为最后一个时间步时
                        elif (i + 1 == len(timesteps_tensor)):
                            s = t  # 保存当前时间步为 s
                            t = prev_timestep  # 更新 t 为前一时间步

                            # 提取 s 和 t 对应的参数
                            lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                            sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                            h = lambda_t - lambda_s
                            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                            phi_1 = torch.expm1(-h)

                            # 使用当前潜变量构造模型输入（考虑 guidance）
                            latent_model_input = torch.cat([latents_aa] * 2) if do_classifier_free_guidance else latents_aa
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            # 调用 UNet 预测噪声，并转换为模型输出
                            noise_pred = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample
                            noise_pred = self.apply_guidance_scale(noise_pred, guidance_scale)
                            model_s = self.scheduler.convert_model_output(noise_pred, t, latents_aa)

                            x_t = latents_aa  # 保存当前状态

                            # 使用公式更新潜变量
                            latents_aa = (sigma_s / sigma_t) * (latents_aa + alpha_t * phi_1 * model_s)

                            # Line 17：若开启优化，则调用固定点校正（order==1）进行修正
                            if (inverse_opt):
                                latents_aa = self.fixedpoint_correction(latents_aa, s, t, x_t, order=1, text_embeddings=text_embeddings, guidance_scale=guidance_scale,
                                                                     step_size=10/t, scheduler=True)   
                        else:
                            raise Exception("Index Error!")
                else:
                    pass

        return latents_aa

    @torch.inference_mode()
    def fixedpoint_correction(self, x, s, t, x_t, r=None, order=1, n_iter=500, step_size=0.1, th=1e-3, 
                                model_s_output=None, model_r_output=None, text_embeddings=None, guidance_scale=3.0, 
                                scheduler=False, factor=0.5, patience=20, anchor=False, warmup=True, warmup_time=20):
        do_classifier_free_guidance = guidance_scale > 1.0
        if order==1:
            input = x.clone()
            original_step_size = step_size
            
            # step size scheduler, reduce when not improved
            if scheduler:
                step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)

            lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
            sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
            h = lambda_t - lambda_s
            phi_1 = torch.expm1(-h)

            for i in range(n_iter):
                # step size warmup
                if warmup:
                    if i < warmup_time:
                        step_size = original_step_size * (i+1)/(warmup_time)
                
                latent_model_input = (torch.cat([input] * 2) if do_classifier_free_guidance else input)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                noise_pred = self.unet(latent_model_input , s, encoder_hidden_states=text_embeddings).sample
                noise_pred = self.apply_guidance_scale(noise_pred, guidance_scale)                   
                model_output = self.scheduler.convert_model_output(noise_pred, s, input)

                x_t_pred = (sigma_t / sigma_s) * input - (alpha_t * phi_1 ) * model_output

                loss = torch.nn.functional.mse_loss(x_t_pred, x_t, reduction='sum')
                
                if loss.item() < th:
                    break                
                
                # forward step method
                input = input - step_size * (x_t_pred- x_t)

                if scheduler:
                    step_size = step_scheduler.step(loss)

            return input        
        
        elif order==2:
            assert r is not None
            input = x.clone()
            original_step_size = step_size
            
            # step size scheduler, reduce when not improved
            if scheduler:
                step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)
            
            lambda_r, lambda_s, lambda_t = self.scheduler.lambda_t[r], self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
            sigma_r, sigma_s, sigma_t = self.scheduler.sigma_t[r], self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
            alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
            h_0 = lambda_s - lambda_r
            h = lambda_t - lambda_s
            r0 = h_0 / h
            phi_1 = torch.expm1(-h)
            
            for i in range(n_iter):
                # step size warmup
                if warmup:
                    if i < warmup_time:
                        step_size = original_step_size * (i+1)/(warmup_time)

                latent_model_input = torch.cat([input] * 2) if do_classifier_free_guidance else input
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample
                noise_pred = self.apply_guidance_scale(noise_pred, guidance_scale) 
                model_output = self.scheduler.convert_model_output(noise_pred, s, input)
                
                x_t_pred = (sigma_t / sigma_s) * input - (alpha_t * phi_1) * model_output
                
                # high-order term approximation
                if i==0:
                    d = (1./ r0) * (model_s_output - model_r_output)
                    diff_term = 0.5 * alpha_t * phi_1 * d

                x_t_pred = x_t_pred - diff_term
                
                loss = torch.nn.functional.mse_loss(x_t_pred, x_t, reduction='sum')

                if loss.item() < th:
                    break                

                # forward step method
                input = input - step_size * (x_t_pred- x_t)

                if scheduler:
                    step_size = step_scheduler.step(loss)
                if anchor:
                    input = (1 - 1/(i+2)) * input + (1/(i+2))*x
            return input
        else:
            raise NotImplementedError

    def decoder_inv(self, x):
        """
        decoder_inv calculates latents z of the image x by solving optimization problem ||E(x)-z||,
        not by directly encoding with VAE encoder. "Decoder inversion"

        INPUT
        x : image data (1, 3, 512, 512)
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z)
        """
        with torch.enable_grad():
            input = x.clone().detach().bfloat16()

            z = self.get_image_latents(x).clone().bfloat16()
            z.requires_grad_(True)

            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam([z], lr=0.1)
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

            for i in self.progress_bar(range(100)):
                x_pred = self.decode_image_for_gradient_float(z)

                loss = loss_function(x_pred, input)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            return z
    
   
        
class StepScheduler(ReduceLROnPlateau):
    def __init__(self, mode='min', current_lr=0, factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        if current_lr == 0:
            raise ValueError('Step size cannot be 0')

        self.min_lr = min_lr
        self.current_lr = current_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            import warnings
            warnings.warn("EPOCH_DEPRECATION_WARNING", UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.current_lr

    def _reduce_lr(self, epoch):
        old_lr = self.current_lr
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.current_lr = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                            "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                        ' to {:.4e}.'.format(epoch_str,new_lr))