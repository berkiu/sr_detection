import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler
from diffusers.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


def preprocess(image):
    if isinstance(image, np.ndarray):
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
    return image


class OnnxStableDiffusionUpscalePipeline(DiffusionPipeline):
    def __init__(
        self,
        vae_decoder: OnnxRuntimeModel,
        unet: OnnxRuntimeModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()

        self.register_modules(
            vae_decoder=vae_decoder,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
        )

    def __call__(
        self,
        image: np.ndarray,
        num_inference_steps: Optional[int] = 75,
        guidance_scale: Optional[float] = 9.0,
        noise_level: Optional[int] = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        
        # 1. Preprocess image
        image = preprocess(image)
        
        if generator is None:
            generator = np.random

        # 2. Define call parameters
        batch_size = 1
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        with open('./text_embeddings.npy', 'rb') as f:
            text_embeddings = np.load(f)
        
        # 4. Check inputs
        image_batch_size = len(image) if isinstance(image, list) else image.shape[0]
        
        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # 5. Add noise to image
        image = image.astype(text_embeddings.dtype)
        noise_level = np.array([noise_level], dtype=text_embeddings.dtype)
        noise = generator.randn(*image.shape).astype(text_embeddings.dtype)
        image = self.low_res_scheduler.add_noise(torch.from_numpy(image), torch.from_numpy(noise), torch.from_numpy(noise_level).long())
        image = image.numpy().astype(text_embeddings.dtype)

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = np.concatenate([image] * batch_multiplier * num_images_per_prompt)
        noise_level = np.concatenate([noise_level] * image.shape[0])

        # 6. Prepare latent variables
        latents_dtype = text_embeddings.dtype
        height, width = image.shape[2:]
        latents_shape = (batch_size * num_images_per_prompt, 4, height, width)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        latents = latents * np.float64(self.scheduler.init_noise_sigma)
           
        # 7. Prepare extra step kwargs.
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
        
        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()
            latent_model_input = np.concatenate([latent_model_input, image], axis=1)

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=text_embeddings, class_labels=noise_level.astype(np.int64))
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.08333 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image * 255
        image = np.expand_dims(image, axis=0)
            
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class StableDiffusionOnnxUpscalePipeline(OnnxStableDiffusionUpscalePipeline):
    def __init__(
        self,
        vae_decoder: OnnxRuntimeModel,
        unet: OnnxRuntimeModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__(
            vae_decoder=vae_decoder,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
        )
