<div align="center">

<h1>
    Upscale-A-Video:<br> 
    Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution
</h1>

<div>
    <a href='https://shangchenzhou.com/' target='_blank'>Shangchen Zhou<sup>∗</sup></a>&emsp;
    <a href='https://pq-yang.github.io/' target='_blank'>Peiqing Yang<sup>∗</sup></a>&emsp;
    <a href='https://iceclear.github.io/' target='_blank'>Jianyi Wang</a>&emsp;
    <a href='https://github.com/Luo-Yihang' target='_blank'>Yihang Luo</a>&emsp;
    <a href='https://www.mmlab-ntu.com/person/ccloy/' target='_blank'>Chen Change Loy</a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>

<div>
    <strong>CVPR 2024 (Highlight)</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://shangchenzhou.com/projects/upscale-a-video/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2312.06640" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.06640-b31b1b.svg">
        </a>
        <a href="https://www.youtube.com/watch?v=b9J3lqiKnLM" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <a href="https://replicate.com/sczhou/upscale-a-video" target='_blank'>
        <img src="https://replicate.com/sczhou/upscale-a-video/badge">
        </a>
        <img src="https://api.infinitescript.com/badgen/count?name=sczhou/Upscale-A-Video">
    </h4>
</div>

<strong>Upscale-A-Video is a diffusion-based model that upscales videos by taking the low-resolution video and text prompts as inputs.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/teaser.png">
</div>

:open_book: For more visual results, go checkout our <a href="##" target="_blank">project page</a>

---
</div>


## 🔥 Update
- [2024.09] Inference code is released.
- [2024.02] YouHQ dataset is made publicly available.
- [2023.12] This repo is created.

## 🎬 Overview
![overall_structure](assets/pipeline.png)

## 🔧 Dependencies and Installation
1. Clone Repo
    ```bash
    git clone https://github.com/sczhou/Upscale-A-Video.git
    cd Upscale-A-Video
    ```

2. Create Conda Environment and Install Dependencies
    ```bash
    # create new conda env
    conda create -n UAV python=3.9 -y
    conda activate UAV

    # install python dependencies
    pip install -r requirements.txt
    ```




## ☕️ Quick Inference

```
 python3 predict.py
```







## 📑 Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @inproceedings{zhou2024upscaleavideo,
      title={{Upscale-A-Video}: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution},
      author={Zhou, Shangchen and Yang, Peiqing and Wang, Jianyi and Luo, Yihang and Loy, Chen Change},
      booktitle={CVPR},
      year={2024}
   }
   ```


## 📝 License

This project is licensed under <a rel="license" href="./LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.


## 📧 Contact
If you have any questions, please feel free to reach us at `shangchenzhou@gmail.com` or `peiqingyang99@outlook.com`. 
