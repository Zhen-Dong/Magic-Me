**Project Page: [Magic-Me](https://magic-me-webpage.github.io/)**

Unlike common text-to-video model (like [OpenAI/Sora](https://openai.com/sora)), this model is for personalized videos using photos of your friends, family, or pets. By training an embedding with these images, it creates custom videos featuring your loved ones, bringing a unique touch to your memories. 

<!-- We provide a library of pre-trained library with 24 characters. You can ask those characters to do anything with text description. You can also customize your own characters such as your friends, families and pets with the provided scripts.-->

**News Update**: We have deployed our model on Hugging Face's GPU platform, making it available for immediate use. Check it out on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/visionMaze/Magic-Me).


# Magic-Me: Identity-Specific Video Customized Diffusion
Ze Ma*, Daquan Zhou* &dagger;, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong &dagger;, Kurt Keutzer, Jiashi Feng 
(*Joint First Author, &dagger; Corresponding Author)
> We propose a new framework for video generation with customized identity. With a pre-trained ID token, the user would be able to generate any video clips with the specified identity. We propose a series of controllable Video generation and editing methods. The first release includes Customized Diffusion (VCD).
It includes three novel components that are essential for high-quality ID preservation: 1) an ID module trained with the cropped identity by prompt-to-segmentation to disentangle the ID information and the background noise for more accurate ID token learning; 2) a text-to-video (T2V) VCD module with 3D Gaussian Noise Prior for better inter-frame consistency and 3) video-to-video (V2V) Face VCD and Tiled VCD modules to deblur the face and upscale the video for higher resolution.

[![arXiv](https://img.shields.io/badge/arXiv-2402.09368-b31b1b.svg)](https://arxiv.org/abs/2402.09368)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gMialn4nkGeDZ72yx1Wob1E1QBgrqeGa?usp=drive_link)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/visionMaze/Magic-Me)
<!-- [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->


**Video Customization Diffusion Model Pipeline**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="__assets__/figs/framwork-comfyui-v1.png" style="width:95%">


**Video Demonstration**

<div align="center">
  <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/8330f571-fecc-45b0-885e-c82682452a6a" width="100">
</div>



**ID Specific Video Generation with reference images**


Altman| Lecun | Robert |Taylor
:-: | :-: | :-: | :-:
| <image width = "500" src=https://github.com/Zhen-Dong/Magic-Me/assets/147695338/ca993c4d-253a-47c1-afcb-4b3716f59f65>  | <image width = "250" src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/dcc611ea-b281-4797-87d9-ff1600f656e0" > | <image width = "500" src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/4345e62c-e59e-4eac-9a0c-0441edfd7c0f"> | <image width = "250" src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/863fb047-bd11-4ee8-a4cd-5e0d9b95ac80"> 
| <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/73f82bd8-c1dd-404e-b7b2-18990119215f">  | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/ba7718ab-0c44-4ef5-9b40-62dd2bf9221d"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/ab6912a8-90bd-4fc8-8b1c-c30a64dcff7a"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/0f169b01-0125-484d-a1c2-34691fdcd165"> |

**ID Specific Video Editing with reference images**

Original Video| Altman | Bengio |Zuck
:-: | :-: | :-: | :-:
| <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/59c097ee-6626-4b6f-9af4-585af46c1d00">  | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/48b9d3bc-65de-4fda-a23d-292e2f00c53d"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/7b3a26d4-4ef4-4c6b-b2ac-a4ad9fb8eec6"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/ad968365-bed0-4df8-9400-9d1b81f709b2"> |
| <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/4bb5c66a-b1d2-40b3-8daa-97b6981e1cba">  | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/8fa83d4f-0a0b-44b2-9686-184f42bba5e3"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/181c0039-85d0-4da3-ab50-29d5cde4d159"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/13412208/6347f651-3451-46a9-96dd-97e5352c10f5"> |

## Next
More works will be released soon. Stay tuned.
1. Magic-Me
    - [ ] Support SD-XL
    - [ ] Integrate pose/depth/stretch control
    - [x] Integrate multi-prompt / Prompt Travel
    - [x] Host model on Hugging Face with Gradio as FE
    - [x] Release arxiv and codes
    - [x] Release demo

2. Magic ID-ditting
    - [ ] Release arxiv and codes
    - [x] Release demo

3. Magic-Me Instant
    - [ ] Release arxiv and codes
    - [ ] Release demo

4. Magic-Me Crowd
    - [ ] Release arxiv and codes
    - [ ] Release demo


## Install 
First, make sure that Anaconda is installed (refer to official [Install Tutorial](https://docs.anaconda.com/free/anaconda/install/)).
```
git clone https://github.com/Zhen-Dong/Magic-Me.git
cd Magic-Me
conda env create -f environment.yaml 
```

## Download checkpoints
1. Stable Diffusion 1.5
   ```
   git lfs install
   git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/
   ```
2. Download motion module and realistic vision checkpoints
   ```
   source download_checkpoint.sh
   ```

## Train and Evaluate Extended ID tokens
- Quick training with evaluation.
    ```
    conda activate magic
    source train.sh
    ```
- Customized training of a specific identity
    1. Select 5-10 images of a specific identity into a new directory under ```dataset```.
    2. Make a new ```.yaml``` out of ```configs/ceo.yaml```.
    3. Run the training codes in conda env **magic**.    
    ```python train.py --config configs/your_config.yaml```
    4. The ID embeddings are available in the directory of **outputs/magic-me-ceo-xxxxx/checkpoints**.
    5. The generated videos by T2V VCD from the prompts in the ```configs/ceo.yaml``` are saved during training in the directory of **outputs/magic-me-ceo-xxxxx/samples**.
       
## Inference: Using Pre-trained Characters to generate video scenes
- Option 1 - Hugging Face
  - We have deployed our pre-trained model on Hugging Face's GPU platform, making it available for immediate use. Check it out on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/visionMaze/Magic-Me).
- Option 2 - Colab
  - This requires you are subscribed to Google Colab. Users must comply with Google Colab's terms of service and respect copyright laws, ensuring that the resources provided are used responsibly and ethically for academic, educational, or research purposes only.
  - Use this Colab Notebook  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gMialn4nkGeDZ72yx1Wob1E1QBgrqeGa?usp=drive_link)  to run ComfyUI.
  - In the ComfyUI, one click on the **"Queue Prompt"** to generate the video.
  - Feel free to change the prompt inside ComfyUI (*embedding:firstname man* for male, *embedding:firstname woman* for female)
    We have provided 24 different character embedding for use:
    
    *altman.pt beyonce.pt harry.pt huang.pt johnson.pt lisa.pt musk.pt taylor.pt andrew_ng.pt biden.pt hermione.pt ironman.pt lecun.pt mona.pt obama.pt trump.pt bengio.pt eli.pt hinton.pt jack_chen.pt lifeifei.pt monroe.pt scarlett.pt zuck.pt*

    <img width="300" alt="image" src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/68e8747f-e7b3-4cb6-afd6-d4ad24f2177d">
  - The available embeddings are cloned into the directory **magic_factory/Magic-ComfyUI/models/embeddings**.

    
    Feel free to put your newly trained embeddings, for example boy1.pt, in the same directory, and mention the mebdding as **embedding:boy1 man** in the ComfyUI prompt.

## BibTeX
```
@misc{ma2024magicme,
      title={Magic-Me: Identity-Specific Video Customized Diffusion}, 
      author={Ze Ma and Daquan Zhou and Chun-Hsiao Yeh and Xue-She Wang and Xiuyu Li and Huanrui Yang and Zhen Dong and Kurt Keutzer and Jiashi Feng},
      year={2024},
      eprint={2402.09368},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Interesting Prompts
**Use embedding:firstname man for the male character and embedding:firstname woman for the female.**
1. a photo of embedding:firstname man in superman costume in the outer space, stars in the background.
2. a photo of embedding:firstname woman, desert, a striking tableau forms around a solitary figure who stoically stands against the whipping winds and stinging sands. Clad in a worn, tattered, dark brown cloak, with hood pulled low, only hinting at the determined visage beneath - this individual embodies fortitude and tenacity.
3. a photo of embedding:firstname woman, clad in exquisite hybrid armor, studded with iridescent gemstones, Cherry blossoms sway gently in the breeze, lithesome figure against a softly blurred backdrop. style of Gerald Parel, ward-winning, professional, highly detailed.
4. a photo of embedding:firstname man walking on the street, B&W, city, captured on a 35mm camera, black and white, classic film. inside Film Still, film grain, Movie Still, Film Still, Cinematic, Cinematic Shot, Cinematic Lighting
5. a photo of embedding:firstname man, amazing quality, knight, holding a sword, dazzling ,transparent ,polishing, waving sword, gold armor, glowing armor, glowing eyes, full armor, shine armor, dazzling armor, masterpiece, best quality, hyper detailed, ultra detailed, UHD, perfect anatomy, (in castle:1.2).


## Disclaimer
This project is released for academic use. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards. 

## Acknowledgements
Codebase built upon [Tune-a-Video](https://github.com/showlab/Tune-A-Video) and [AnimateDiff](https://github.com/guoyww/AnimateDiff)
