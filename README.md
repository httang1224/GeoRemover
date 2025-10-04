<div align="center">
  <h1><b>GeoRemover: Removing Objects and Their Causal Visual Artifacts</b></h1>
  <h3>NeurIPS 2025 (Spotlight)</h3>
  <p>
    <a href="https://arxiv.org/abs/2509.18538"><b>Paper (arXiv)</b></a> ·
    <a href="https://huggingface.co/spaces/buxiangzhiren/GeoRemover"><b>🚀 Try the Demo </b></a>
  </p>
  <p>
    Zixin Zhu, Haoxiang Li, Xuelu Feng, He Wu, Chunming Qiao, Junsong Yuan
  </p>
  <img src="docs/teaser.png" width="800"/>
</div>

---

> **Abstract.** *Towards intelligent image editing, object removal should eliminate both the target object and its causal visual artifacts, such as shadows and reflections. However, existing image appearance-based methods either follow strictly mask-aligned training and fail to remove these casual effects which are not explicitly masked, or adopt loosely mask-aligned strategies that lack controllability and may unintentionally over-erase other objects. We identify that these limitations stem from ignoring the causal relationship between an object’s geometry presence and its visual effects. To address this limitation, we propose a geometry-aware two-stage framework that decouples object removal into (1) geometry removal and (2) appearance rendering. In the first stage, we remove the object directly from the geometry (e.g., depth) using strictly mask-aligned supervision, enabling structure-aware editing with strong geometric constraints. In the second stage, we render a photorealistic RGB image conditioned on the updated geometry, where causal visual effects are considered implicitly as a result of the modified 3D geometry. To guide learning in the geometry removal stage, we introduce a preference-driven objective based on positive and negative sample pairs, encouraging the model to remove objects as well as their causal visual artifacts while avoiding new structural insertions. Extensive experiments demonstrate that our method achieves state-of-the-art performance in removing both objects and their associated artifacts on two popular benchmarks.*

---

## 🔧 Installing the Dependencies

Before running scripts locally, install environment requirements:

```bash
bash env.sh
```

Then initialize 🤗 Accelerate:

```bash
accelerate config
# or
accelerate config default
```

---

## 📦 Data Preparation

- Download images from **[RORD](https://github.com/Forty-lock/RORD)**.
- Generate depth maps with **[Video-Depth-Anything v2](https://github.com/DepthAnything/Video-Depth-Anything)**.  
  The depth code is available in this repo (see the `depth` branch) and can be run via:
  - Script: **[run_images_rord.py](https://github.com/buxiangzhiren/GeoRemover/blob/depth/run_images_rord.py)**

---

## 🏋️ Training

First, build your own `train_images_and_rord_masks.csv`.  
(The CSV included here is a small example, not the full RORD.)

**Stage-1: geometry removal**
```bash
bash train_stage1.sh
```

**Stage-2: appearance rendering (REQUIRED)**
```bash
bash train_stage2.sh
```

---

## 🔎 Inference

1) First compute depth for your image(s):  
   **[run_single_image.py](https://github.com/buxiangzhiren/GeoRemover/blob/depth/run_single_image.py)**

2) **Stage-1: geometry removal**
```bash
python Flux_fill_infer_depth.py
```

3) **Stage-2: appearance rendering (REQUIRED)**
```bash
python Flux_fill_d2i.py
```

> **Note:** Stage-2 is not optional. It is the depth→image rendering step that produces the final photorealistic result conditioned on the updated geometry from Stage-1.

---

## 🧰 Checkpoints

- **Hugging Face:**  
  **[Stage-1 (geometry removal) & Stage-2 (appearance rendering)](https://huggingface.co/buxiangzhiren/GeoRemover)**

- **Google Drive:**  
  **[Stage-1: geometry removal](https://drive.google.com/file/d/1y6vnxqnFTiO6sxoKDBkvFbAeniHFka89/view?usp=sharing)**  
  **[Stage-2: appearance rendering](https://drive.google.com/file/d/1U8rp1hqOswQB-0T0fh2aDQu-o1GLfd6E/view?usp=sharing)**

---

## 🙏 Acknowledgements

This repo builds on:
- **[RORD](https://github.com/Forty-lock/RORD)**
- **[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)**
- **[Video-Depth-Anything v2](https://github.com/DepthAnything/Video-Depth-Anything)**

Thanks for their excellent work!

---

## 📚 Citation

```bibtex
@misc{zhu2025georemoverremovingobjectscausal,
      title={GeoRemover: Removing Objects and Their Causal Visual Artifacts}, 
      author={Zixin Zhu and Haoxiang Li and Xuelu Feng and He Wu and Chunming Qiao and Junsong Yuan},
      year={2025},
      eprint={2509.18538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.18538}, 
}
```
