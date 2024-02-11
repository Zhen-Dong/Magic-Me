# Magic-Me

This repo includes a series of research work about ID-specific content generation by UC Berkeley X ByteDance.
[![Project Page](https://img.shields.io/badge/Project-Website-green)]()
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**Magic Album: Identity-Specific Video Customized Diffusion**
</br>
Ze Ma*, Daquan Zhou* &dagger;, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong &dagger;, Kurt Keutzer, Jiashi Feng 
(*Joint First Author, &dagger; Corresponding Author)


- **Explanation:** Video Customized Diffusion (VCD) injects frame-wise correlation at the noise initialization for stable video outputs and reinforces the identity information at different VCD stages.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="__assets__/figs/framwork-comfyui-v1.png" style="width:95%">

Altman as a superman | caption2 | Altman as a superman |caption2
:-: | :-: | :-: | :-:
<video src='__assets__/videos/altman-superman.mp4' width=180/> | <video src='__assets__/videos/ironman-desert.mp4' width=180/> | <video src='__assets__/videos/ironman-desert.mp4' width=180/> | <video src='__assets__/videos/altman-walking.mp4' width=180/>


<div style="display: flex; justify-content: space-between;">
  <video controls="true" autoplay="true" loop="true" style="width: 24%;"><source src="__assets__/videos/altman-superman.mp4" type="video/mp4"></video>
  <video controls="true" autoplay="true" loop="true" style="width: 24%;"><source src="__assets__/videos/ironman-desert.mp4" type="video/mp4"></video>
  <video controls="true" autoplay="true" loop="true" style="width: 24%;"><source src="__assets__/videos/taylor-light.mp4" type="video/mp4"></video>
  <video controls="true" autoplay="true" loop="true" style="width: 24%;"><source src="__assets__/videos/altman-walking.mp4" type="video/mp4"></video>
</div>


## Next
More works will be released soon. Stay tuned.
1. Magic Album
    - [ ] Support SD-XL
    - [ ] Integrate pose/depth/stretch control
    - [x] Integrate multi-prompt / Prompt Travel
    - [x] Release arxiv and codes
    - [x] Release demo

2. Magic Editting
    - [ ] Release arxiv and codes
    - [x] Release demo

3. Magic Album Zero
    - [ ] Release arxiv and codes
    - [ ] Release demo

4. Magic Album People
    - [ ] Release arxiv and codes
    - [ ] Release demo


## Install 
```
git lfs install
git clone https://github.com/Zhen-Dong/Magic-Me.git
cd Magic-Me
conda env create -f environment.yaml 
source download_checkpoint.sh
```

## Train extended ID tokens
- Quick training.
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
- Magic Album: Click the link of colab and follow the instructions [![Colab](https://colab.research.google.com/assets/colab-badge.svg)]().

<!-- ## BibTeX
```
@article{
}
``` -->

## Disclaimer
This project is released for academic use. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards. 

## Acknowledgements
Codebase built upon [Tune-a-Video](https://github.com/showlab/Tune-A-Video) and [AnimateDiff](https://github.com/guoyww/AnimateDiff)