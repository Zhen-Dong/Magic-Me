import os
import math
import wandb
import logging
import inspect
import argparse
import datetime
import sys
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import animatediff.pipelines.vcd_pipeline_animation as vcd_pipeline_animation

from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.data.masked_dataset import MaskedCustomDataset
from animatediff.models.unet import UNet3DConditionModel
from animatediff.utils.util import save_videos_grid, zero_rank_print, freeze_and_add_custom_token, add_token, save_embedding
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint



class Logger(object):
    def __init__(self, f):
        self.terminal = sys.stdout
        self.log = open(f, "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.terminal.flush()
        self.log.flush()  
   


def main(
    config_name: str,
    use_wandb: bool,
    launcher: str,
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    dreambooth_model_path: str = "",
    prefix: str = "",
    sub_name: str = "",
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {"use_motion_module": False,   
                                    "unet_use_cross_frame_attention": False,
                                    "unet_use_temporal_attention": False},
    noise_scheduler_kwargs = None,
    modifier_tokens = ["<new1>","<new2>","<new3>","<new4>"],
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    num_workers: int = 4,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
    noise_offset: float = 0.0,    
):
    check_min_version("0.10.0.dev0")

    local_rank = 0
    global_rank = 0
    num_processes = 1
    is_main_process = 1

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else prefix + config_name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S") + sub_name
    output_dir = os.path.join(output_dir, folder_name)

    # Handle the output folder creation
    *_, config = inspect.getargvalues(inspect.currentframe())
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    double_loggger = Logger(os.path.join(output_dir, 'train.log'))
    sys.stdout = double_loggger
    sys.stderr = double_loggger


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout
    )

    # if is_main_process and (not is_debug) and use_wandb:
    if is_main_process and use_wandb:
        _ = wandb.init(project="animatecustom", name=folder_name, config=config)


    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=unet_additional_kwargs)

    video_vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    video_tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    video_text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    video_unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(validation_data.unet_additional_kwargs))
    motion_module_state_dict = torch.load(validation_data.motion_module, map_location="cpu")
    missing, unexpected = video_unet.load_state_dict(motion_module_state_dict, strict=False)
    print(f"load inferece motion module missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if dreambooth_model_path != "":
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
            converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, unet.config)

            missing, unexpected = unet.load_state_dict(converted_unet_checkpoint, strict=False)
            print(f"load dreambooth_model_path unet missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            assert len(unexpected) == 0

            text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict, text_encoder)
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, vae.config)
            missing, unexpected = vae.load_state_dict(converted_vae_checkpoint, strict=False)
            print(f"load dreambooth_model_path vae missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

        else:
            raise NotImplementedError()


    if validation_data.path != "":
        if validation_data.path.endswith(".safetensors"):
            inference_state_dict = {}
            with safe_open(validation_data.path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    inference_state_dict[key] = f.get_tensor(key)
            converted_unet_checkpoint = convert_ldm_unet_checkpoint(inference_state_dict, video_unet.config)

            missing, unexpected = video_unet.load_state_dict(converted_unet_checkpoint, strict=False)
            print(f"load inferece unet missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            assert len(unexpected) == 0

            video_text_encoder = convert_ldm_clip_checkpoint(inference_state_dict, video_text_encoder)
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(inference_state_dict, video_vae.config)
            missing, unexpected = video_vae.load_state_dict(converted_vae_checkpoint, strict=False)
            print(f"load inferece vae missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            # assert len(unexpected) == 0
            add_token(video_tokenizer, video_text_encoder, modifier_tokens)


        else:
            raise NotImplementedError()
    else:
        add_token(video_tokenizer, video_text_encoder, modifier_tokens)


    video_unet.eval()

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_state_dict = {}
        with safe_open(unet_checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                unet_state_dict[key] = f.get_tensor(key)
        if "global_step" in unet_state_dict: zero_rank_print(f"global_step: {unet_state_dict['global_step']}")
        unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, unet.config)
        m, u = unet.load_state_dict(unet_state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        
    # Freeze vae and text_encoder
    _ = freeze_and_add_custom_token(tokenizer, text_encoder, modifier_tokens)
    vae.requires_grad_(False)
    vae.eval()
    unet.requires_grad_(False)


    # Enable xformers
    # Must be called before get the unet.attn_processors 
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            video_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    for _, attn in unet.attn_processors.items():
        if isinstance(attn, torch.nn.Module):
            attn.requires_grad_(False)
                
                            
    trainable_params = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    print(f"trainable params: {len(trainable_params)}")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )   
    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")


    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = MaskedCustomDataset(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    # if not image_finetune:
    video_validation_pipeline = vcd_pipeline_animation.AnimationPipeline(
        unet=video_unet, vae=video_vae, tokenizer=video_tokenizer, text_encoder=video_text_encoder, 
        scheduler=DDIMScheduler(**OmegaConf.to_container(validation_data.noise_scheduler_kwargs))
    ).to("cuda")
    video_validation_pipeline.enable_vae_slicing()



    unet.to(local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    len_train_dataloader = len(train_dataloader) if len(train_dataloader) > 0 else train_batch_size
    num_update_steps_per_epoch = math.ceil(len_train_dataloader/ gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        
        
        for step, batch in enumerate(train_dataloader):
            unet.train()
                
            # Data batch sanity check
            if global_step % checkpointing_steps == 0:
                masks, pixel_values, texts = batch['masks'].cpu(), batch['pixel_values'].cpu(), batch['text']
                masks = masks.float()     
                masks = torch.nn.Upsample(scale_factor=(8, 8))(masks.squeeze(1)).unsqueeze(1)
                new_pixel_values = pixel_values * masks
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                new_pixel_values = rearrange(new_pixel_values, "b f c h w -> b c f h w")
                masks = rearrange(masks, "b f c h w -> b c f h w")
                for idx, (pixel_value, new_pixel_value, masks, text) in enumerate(zip(pixel_values, new_pixel_values, masks, texts)):
                    pixel_value = pixel_value[None, ...]
                    new_pixel_value = new_pixel_value[None, ...]
                    masks = masks[None, ...]
                    save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{global_rank}-{global_step}-{idx}-{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else 'non-prompt'}-pixel.gif", rescale=True)
                    save_videos_grid(new_pixel_value, f"{output_dir}/sanity_check/{global_rank}-{global_step}-{idx}-{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else 'non-prompt'}-masked-pixel.gif", rescale=True)
                    save_videos_grid(masks, f"{output_dir}/sanity_check/{global_rank}-{global_step}-{idx}-{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else 'non-prompt'}-mask.gif", rescale=True)
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            noise += noise_offset * torch.randn(latents.shape[0], latents.shape[1], *([1]*(len(latents.shape)-2)), device=latents.device)
            bsz = latents.shape[0]
            
            
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompts = batch['text']
                prompt_ids = tokenizer(
                   prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)

            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                # Predict the noise residual and compute loss
                # Mixed-precision training
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise.detach()
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                masks = batch['masks'].to(local_rank)
                # torch.seed()
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float()*masks, target.float()*masks, reduction="none")
                loss = loss.sum([1,2,3,4])/masks.sum([1,2,3,4])
                loss = loss.mean()


            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and global_step % checkpointing_steps == 0 and global_step != 0:
                save_path = os.path.join(output_dir, f"checkpoints")
                embeds = text_encoder.state_dict()["text_model.embeddings.token_embedding.weight"]
                save_embedding(embeds, len(modifier_tokens), save_path, f'{config_name}-gstep-{global_step}.pt' )
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 and global_step != 0):
                unet.eval()
                video_samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size
    
                prompts = validation_data.prompts
                n_prompts    = list(validation_data.n_prompts) * len(validation_data.prompts) if len(validation_data.prompts) == 1 else validation_data.n_prompts
                random_seeds = validation_data.get("seed", [-1])
                random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
                random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
                video_validation_pipeline.text_encoder.load_state_dict(text_encoder.state_dict())
                batch_sz = min(len(prompts), 4)
                batch_idx = 0
                start_idx = batch_idx*batch_sz
                end_idx = min((batch_idx+1)*batch_sz, len(prompts))

                for batch_idx in range(math.ceil(len(prompts) / batch_sz)):
                    start_idx = batch_idx*batch_sz
                    end_idx = min((batch_idx+1)*batch_sz, len(prompts))
                    with torch.no_grad():
                        video_sample = video_validation_pipeline(
                                list(prompts)[start_idx:end_idx],
                                negative_prompt     = list(n_prompts)[start_idx:end_idx],
                                generator    = generator,
                                height       = height,
                                width        = width,
                                modifier_tokens = modifier_tokens,
                                **validation_data,
                        ).videos
                        video_samples.append(video_sample)
                                                        
                video_samples = torch.concat(video_samples)
                save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                save_videos_grid(video_samples, save_path)
                logging.info(f"Saved samples to {save_path}")
                if use_wandb:
                    wandb.log({"video": wandb.Video(save_path, caption="|".join(list(prompts)), fps=8)}, step=global_step)
                                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    save_path = os.path.join(output_dir, f"checkpoints")
    embeds = text_encoder.state_dict()["text_model.embeddings.token_embedding.weight"]
    save_embedding(embeds, len(modifier_tokens), save_path, 'last.pt' )
    logging.info(f"Saved state to {save_path} (global_step: {global_step})")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--prefix",   type=str, default="")
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(config_name=name, prefix=args.prefix, launcher=args.launcher, use_wandb=args.wandb, **config)
