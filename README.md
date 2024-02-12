**Project Page: [Magic-Me]()**

# Magic-Me
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arxiv-soon_to_be_released-blue)]()


Magic-Me includes a series of research work about ID-specific content generation by UC Berkeley and ByteDance.


**Magic-Me: Identity-Specific Video Customized Diffusion**
</br>
Ze Ma*, Daquan Zhou* &dagger;, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong &dagger;, Kurt Keutzer, Jiashi Feng 
(*Joint First Author, &dagger; Corresponding Author)


- **Explanation:** Video Customized Diffusion (VCD) injects frame-wise correlation at the noise initialization for stable video outputs and reinforces the identity information at different VCD stages.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="__assets__/figs/framwork-comfyui-v1.png" style="width:95%">

**ID Specific Video Generation**

Altman| Altman | Robert |Taylor
:-: | :-: | :-: | :-:
| <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/6f18f2f9-89e5-4c05-9f3c-723bc374e9de">  | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/454497b3-f0bf-4341-9415-e068f39e9246"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/ab6912a8-90bd-4fc8-8b1c-c30a64dcff7a"> | <video src="https://github.com/Zhen-Dong/Magic-Me/assets/147695338/0f169b01-0125-484d-a1c2-34691fdcd165"> |




**ID Specific Video Editing**

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

## Train and evaluate extended ID tokens
- Quick training with evaluation.
    ```
    conda activate magic
    source train.sh
    ```
- Customized training of a specific identity
    1. Select 5-10 images of a specific identity into a new diretory under ```dataset```.
    2. Make a new ```.yaml``` out of ```configs/ceo.yaml```.
    3. Run the training codes in conda env **magic**.    
    ```python train.py --config configs/your_config.yaml```

## Inference
- Magic Album:
  - use this Colab Notebook  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12mcMlWjMm3d7etovSCC1CxU4lk2d1bno?usp=sharing)  to run ComfyUI with cloudflared
  - By following the instruction in the notebook, you will be able to use our pre-trained embeddings (incl. Sam Altman, Harry Potter, Andrew Ng, etc.) to generate your own Text-to-Video.
  - Apart from the default ComfyUI workflow (which will be automatically preloaded when you open ComfyUI the first time), we also provide the following workflows using different prompts to help you explore more scenes:
    - superman.json
    - cherry_blossoms.json
    - walking.json
    - northern_lights.json
    - desert.json


<!-- ## BibTeX
```
@article{
}
``` -->

## Interesting Prmopts:
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
