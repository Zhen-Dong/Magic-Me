
import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange, repeat
import random
from lang_sam import LangSAM
from animatediff.utils.util import save_videos_grid
import torchvision.transforms.functional as TF
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

templates_small = [
    'photo of a {}',
]


templates_small_style = [
    'painting in the style of {}',
]

image_suffix = ['png' , 'JPG', 'jpg', 'jpeg', 'webp']

def isimage(path):
    for suffix in image_suffix:
        if suffix in path.lower():
            return True
    return False

def atoi(text):
    return int(text) if text.isdigit() else text

def index_key(path):
    images_suffix_regex_patten = '|'.join(image_suffix)
    return int(re.match(rf'.*(\d+)\.({images_suffix_regex_patten})', os.path.basename(path)).group(1))


class MaskedCustomDataset(Dataset):
    def __init__(self,
                 datapath,
                 reg_datapath=None,
                 caption=None,
                 sam_prompt=None,
                 reg_caption=None,
                 sample_size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 use_orig_img_rate = 0.1,
                 aug=True,
                 style=False,
                 repeat_times=0.,
                 rotate_p = 0.3,
                 max_rotate_angle = 10,
                 ):

        self.aug = aug
        self.repeat_times = repeat_times
        self.style = style
        self.templates_small = templates_small
        self.use_orig_img_rate = use_orig_img_rate
        self.caption = caption
        self.flip_p = flip_p
        self.rotate_p = rotate_p
        self.max_rotate_angle = max_rotate_angle
        # self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.size = sample_size
        self.interpolation = {
                                # "linear": PIL.Image.LINEAR,
                              "bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]


        if self.caption.endswith('txt'):
            if os.path.exists(self.caption):
                self.caption = [x.strip() for x in open(caption, 'r').readlines()]
            else:
                raise FileNotFoundError()
        
        self.reg_caption = reg_caption
        if self.reg_caption is not None and os.path.exists(self.reg_caption):
            self.reg_caption = [x.strip() for x in open(reg_caption, 'r').readlines()]

        
        
        if self.style:
            self.templates_small = templates_small_style
        if os.path.isdir(datapath):
            self.image_paths1 = [os.path.join(datapath, file_path) for file_path in os.listdir(datapath) if isimage(file_path)]
            self.image_paths1 = sorted(self.image_paths1, key=index_key)
        else:
            with open(datapath, "r") as f:
                self.image_paths1 = f.read().splitlines()

        self._length1 = len(self.image_paths1)
        
        self.image_paths2 = []
        self._length2 = 0
        if reg_datapath is not None:
            if os.path.isdir(reg_datapath):
                self.image_paths2 = [os.path.join(reg_datapath, file_path) for file_path in os.listdir(reg_datapath) if isimage(file_path)]
            else:
                with open(reg_datapath, "r") as f:
                    self.image_paths2 = f.read().splitlines()
            self._length2 = len(self.image_paths2)
        model = LangSAM()
        
        self.images = {'masks': [], 'masked_images': [], 'orig_images': [], 'orig_reg_images': [], 'masked_reg_images': []}
        
        for img_path in self.image_paths1: 
            try:
                image = Image.open(img_path).convert('RGB')
            except OSError as e:
                print(f"Error occurred with image file: {img_path}")
                # Re-throwing a new error or the same error, depending on your requirement
                raise e
            if sam_prompt is None:
                raise NotImplementedError()
            print("sam_prmopt: ", sam_prompt)
            masks, boxes, phrases, logits = model.predict(image, sam_prompt)
            if masks[0,:,:].cpu().numpy().sum() < 1:
                print(f"ERROR: the image {img_path} receives empty mask, skip")
                continue
            img_orig = np.array(image)
            mask = repeat(masks[0, :, :], 'h w -> h w c', c=img_orig.shape[-1]).numpy().astype(int)
            mask = mask.astype(np.uint8)
            img_masked = img_orig * mask
            self.images['masked_images'].append(img_masked)
            self.images['orig_images'].append(img_orig)
            self.images['masks'].append(mask)

            
        if not isinstance(self.caption, str):
            assert len(self.caption) == len(self.images['orig_images']), f"len {len(self.caption)} != len {len(self.images['orig_images'])}"
        del model


    def __len__(self):
        if self._length2 > 0:
            return 2*self._length2
        elif self.repeat_times > 0:
            return self._length1*self.repeat_times
        else:
            return self._length1

    def __getitem__(self, i):
        example = {}
        if i > self._length2 or self._length2 == 0:
            if isinstance(self.caption, str):
                example["text"] = np.random.choice(self.templates_small).format(self.caption)
            else:
                example["text"] = self.caption[i % min(self._length1, len(self.caption)) ]

            if random.uniform(0, 1) <= self.use_orig_img_rate:
                image = self.images['orig_images'][i % self._length1]
                mask = np.ones(image.shape)
            else:
                image = self.images['orig_images'][i % self._length1]
                mask = self.images['masks'][i % self._length1]
                # example["text"] = example["text"].split("scene:")[0].strip()
        else:
            raise NotImplementedError()
        
        # default to score-sde preprocessing
        img = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]

        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        mask = mask[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        cropped_orig_img = Image.fromarray(img)
        scaled_orig_img = cropped_orig_img.resize((256, 256))
        example['cropped_orig_img'] = np.array(scaled_orig_img).astype(np.uint8) # (h, w, c)

        image = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if random.random() > self.flip_p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > self.rotate_p:
            angle =  np.random.normal(scale=self.max_rotate_angle)
            image = TF.rotate(image, angle=angle)
            mask = TF.rotate(mask, angle=angle)

        if i > self._length2 or self._length2 == 0:
            random_scale = self.size
            if self.aug:
                if np.random.randint(0, 3) < 2:
                    random_scale = np.random.randint(self.size // 3, self.size+1)
                else:
                    random_scale = np.random.randint(int(1.2*self.size), int(1.4*self.size))

                if random_scale % 2 == 1:
                    random_scale += 1
            else:
                random_scale = self.size

            if random_scale < 0.6*self.size:
                add_to_caption = np.random.choice(["a far away ", "very small "])
                example["text"] = add_to_caption + example["text"]
                cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
                cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

                image = image.resize((random_scale, random_scale), resample=self.interpolation)
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)

                mask = mask.resize((random_scale, random_scale), resample=self.interpolation)
                mask = np.array(mask).astype(np.uint8)


                input_image1 = np.zeros((self.size, self.size, 3), dtype=np.float32)
                input_image1[cx - random_scale // 2: cx + random_scale // 2, cy - random_scale // 2: cy + random_scale // 2, :] = image

                input_mask1 = np.zeros((self.size, self.size, 3), dtype=np.uint8)
                input_mask1[cx - random_scale // 2: cx + random_scale // 2, cy - random_scale // 2: cy + random_scale // 2, :] = mask
                mask = Image.fromarray(input_mask1).resize((self.size//8, self.size//8), self.interpolation)
                mask = np.array(mask)

            elif random_scale > self.size:
                add_to_caption = np.random.choice(["zoomed in ", "close up "])
                example["text"] = add_to_caption + example["text"]
                cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
                cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
                
                image = image.resize((random_scale, random_scale), resample=self.interpolation)
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)
                mask = mask.resize((random_scale, random_scale), resample=self.interpolation)
                mask = np.array(mask).astype(np.uint8)
                
                
                input_image1 = image[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]
                input_mask1 = mask[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]
                mask = Image.fromarray(input_mask1).resize((self.size//8, self.size//8), self.interpolation)
                mask = np.array(mask)
            else:
                if self.size is not None:
                    image = image.resize((self.size, self.size), resample=self.interpolation)
                input_image1 = np.array(image).astype(np.uint8)
                input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
                mask = mask.resize((self.size//8, self.size//8), self.interpolation)
                mask = np.array(mask)
                

        else:
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)
            input_image1 = np.array(image).astype(np.uint8)
            input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
            mask = Image.fromarray(mask).resize((self.size//8, self.size//8), self.interpolation)
            mask = np.array(mask)
            
        mask[mask > 1] = 1
        example["pixel_values"] = repeat(torch.from_numpy(input_image1), 'h w c -> f c h w', f=1) # (f, c, h, w)
        example["masks"] = repeat(torch.from_numpy(mask), 'h w c -> f c h w', f=1) # (f, c, h, w)
        return example
    
if __name__ == "__main__":
    import torch
    dataset = MaskedCustomDataset(
        datapath =   "dataset/benchmark_dataset/person_3",
        sam_prompt = "male man",
        caption= "dataset/person_3_caption.txt",
        sample_size=  512,
        use_orig_img_rate= 0.5,
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        masks, pixel_values, texts = batch['masks'].cpu(), batch['pixel_values'].cpu(), batch['text']
        masks = masks.float()     
        masks = torch.nn.Upsample(scale_factor=(8, 8), mode='bicubic')(masks.squeeze(1)).unsqueeze(1)
        new_pixel_values = pixel_values * masks
        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        new_pixel_values = rearrange(new_pixel_values, "b f c h w -> b c f h w")
        mask = rearrange(masks, "b f c h w -> b c f h w")
        mask[mask > 1] = 1
        for idx, (pixel_value, new_pixel_value, mask, text) in enumerate(zip(pixel_values, new_pixel_values, masks, texts)):
            pixel_value = pixel_value[None, ...]
            new_pixel_value = new_pixel_value[None, ...]
            mask = mask[None, ...]
            save_videos_grid(pixel_value, f"./{idx}-{text[-30:]}-pixel.gif", rescale=True)
            save_videos_grid(new_pixel_value, f"./{idx}-{text[-30:]}-new_pixel.gif", rescale=True)
            save_videos_grid(mask, f"./{idx}-{text[-30:]}-mask.gif", rescale=True)
