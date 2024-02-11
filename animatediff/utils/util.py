import os
import imageio
import numpy as np
from typing import Union, List

import torch
import torchvision
import torch.distributed as dist

from tqdm import tqdm
from einops import rearrange
import PIL.Image
import PIL.ImageOps
import transformers
from packaging import version
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Any, Optional, Tuple, Union

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def save_image(imgs: torch.Tensor, path: str, rescale=False):
    imgs = rearrange(imgs, "b c f h w -> b f h w c")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for idx, frames in enumerate(imgs):
        for f, img in enumerate(frames):
            if rescale:
                img = (img + 1.0) / 2.0  # -1,1 -> 0,1
            img = (img * 255).numpy().astype(np.uint8)
            Image.fromarray(img).save(f"{path}_{idx}_{f}.jpeg")
            


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def pt_to_pil(images):
    """
    Convert a torch image to a PIL image.
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = numpy_to_pil(images)
    return images


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]


def preprocess_image(image, size=512):
    if not image.mode == "RGB":
        image = image.convert("RGB")
    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]

    img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

    image = Image.fromarray(img)
    image = image.resize((size, size), resample=PIL.Image.BICUBIC)

    image = np.array(image)[None, :]
    image = image / 127.5 - 1
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)
    
def freeze_and_add_custom_token(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, modifier_tokens=['<new1>','<new2>','<new3>','<new4>']):
    text_encoder.eval()
    for param in text_encoder.text_model.encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.text_model.final_layer_norm.parameters():
        param.requires_grad = False
    for param in text_encoder.text_model.embeddings.position_embedding.parameters():
        param.requires_grad = False
        
        
    modifier_token_id_list = add_token(tokenizer, text_encoder, modifier_tokens)
    
    def forward(
        self: CLIPTextModel,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Any:
        indices = input_ids == modifier_token_id_list[-1]
        for token_id in modifier_token_id_list:
            indices |= input_ids == token_id
        indices = (indices*1).unsqueeze(-1)

        output_attentions = output_attentions if output_attentions is not None else self.text_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.text_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.text_model.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)
        hidden_states = (1-indices)*hidden_states.detach() + indices*hidden_states
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        bsz, seq_len = input_shape[:2]
        if version.parse(transformers.__version__) >= version.parse('4.21'):
            causal_attention_mask = self.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )
        else:
            causal_attention_mask = self.text_model._build_causal_attention_mask(bsz, seq_len).to(
                hidden_states.device
            )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)


        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device), input_ids.to(torch.int).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    bound_forward = forward.__get__(text_encoder, CLIPTextModel)
    setattr(text_encoder, 'forward', bound_forward)
    return modifier_token_id_list

def add_token(tokenizer, text_encoder: CLIPTextModel, modifier_token=['<new1>','<new2>','<new3>']):
    modifier_token_id_list = []

    for each_modifier_token in modifier_token:
        num_added_tokens = tokenizer.add_tokens(each_modifier_token)
        modifier_token_id = tokenizer.convert_tokens_to_ids(each_modifier_token)
        modifier_token_id_list.append(modifier_token_id)

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[modifier_token_id_list[-1]] = torch.nn.Parameter(token_embeds[42170])
    for idx, token in enumerate(modifier_token):
        if token == '<new1>':
            token_embeds[modifier_token_id_list[idx]] = torch.nn.Parameter(token_embeds[42170])
        elif token == '<new2>':
            token_embeds[modifier_token_id_list[idx]] = torch.nn.Parameter(token_embeds[47629])
        elif token == '<new3>':
            token_embeds[modifier_token_id_list[idx]] = torch.nn.Parameter(token_embeds[43514])
        else:
            token_embeds[modifier_token_id_list[idx]] = torch.nn.Parameter(token_embeds[43514])

    return modifier_token_id_list

def load_motion_lora(pipeline, state_dict, alpha=1.0):
    # directly update weight in diffusers model
    for key in state_dict:
        # only process lora down key
        if "up." in key: continue

        up_key    = key.replace(".down.", ".up.")
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        model_key = model_key.replace("to_out.", "to_out.0.")
        layer_infos = model_key.split(".")[:-1]

        curr_layer = pipeline.unet
        while len(layer_infos) > 0:
            temp_name = layer_infos.pop(0)
            curr_layer = curr_layer.__getattr__(temp_name)

        weight_down = state_dict[key]
        weight_up   = state_dict[up_key]
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)

    return pipeline
    

def save_embedding(embed, num_embeddings, save_directory, filename, start_idx=49408):
    """
    Create a multi-embedding tensor and save it as a .pt file.
    """
    embeddings = torch.Tensor(embed[start_idx:start_idx+num_embeddings].cpu().numpy())
    tensor_dict = {"string_to_param": {'*': embeddings}}
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    torch.save(tensor_dict, os.path.join(save_directory, filename))
