The official implementation of the **NeurIPS 2025** paper:

<div align="center">
<h1>
<b>
GeoRemover: Removing Objects and Their Causal Visual Artifacts, NeurIPS, 2025 (Spotlight) 
</b>
</h1>
</div>

<p align="center"><img src="docs/teaser.png" width="800"/></p>

> [**GeoRemover: Removing Objects and Their Causal Visual Artifacts**](https://arxiv.org/abs/2509.18538)
> 
> Zixin Zhu, Haoxiang Li, Xuelu Feng, He Wu, Chunming Qiao, Junsong Yuan 

> **Abstract:** *Towards intelligent image editing, object removal should eliminate both the target object and its causal visual artifacts, such as shadows and reflections. However, existing image appearance-based methods either follow strictly mask-aligned training and fail to remove these casual effects which are not explicitly masked, or adopt loosely mask-aligned strategies that lack controllability and may unintentionally over-erase other objects. We identify that these limitations stem from ignoring the causal relationship between an object’s geometry presence and its visual effects. To address this limitation, we propose a geometry-aware two-stage framework that decouples object removal into (1) geometry removal and (2) appearance rendering. In the first stage, we remove the object directly from the geometry (e.g., depth) using strictly mask-aligned supervision, enabling structure-aware editing with strong geometric constraints. In the second stage, we render a photorealistic RGB image conditioned on the updated geometry, where causal visual effects are considered implicitly as a result of the modified 3D geometry. To guide learning in the geometry removal stage, we introduce a preference-driven objective based on positive and negative sample pairs, encouraging the model to remove objects as well as their causal visual artifacts while avoiding new structural insertions. Extensive experiments demonstrate that our method achieves state-of-the-art performance in removing both objects and their associated artifacts on two popular benchmarks.*

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

```bash
bash env.sh
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

### Data prepare
Download the images on [RORD](https://github.com/Forty-lock/RORD) and generate depth maps with [Video-Depth-Anythingv2](https://github.com/DepthAnything/Video-Depth-Anything). (The code for VideoDepthAnything v2 can be found in the same repository, on the `depth` branch, using the [script](https://github.com/buxiangzhiren/GeoRemover/blob/depth/run_images_rord.py))

### Training
You should build your own *train_images_and_rord_masks.csv* first. The file in the repo is not the full RORD—it's just an example.

For stage1:geometry removal
```bash
bash train_stage1.sh
```
For stage2:appearance rendering
```bash
bash train_stage2.sh
```
### Inference
First, use https://github.com/buxiangzhiren/GeoRemover/blob/depth/run_single_image.py to get the depth of a image

For stage1:geometry removal
```bash
python Flux_fill_infer_depth.py
```
For stage2:appearance rendering
```bash
python Flux_fill_d2i.py
```
### Checkpoints
Hugging Face:
[stage1:geometry removal and stage2:appearance rendering](https://huggingface.co/buxiangzhiren/GeoRemover)


Google drive:
[stage1:geometry removal](https://drive.google.com/file/d/1y6vnxqnFTiO6sxoKDBkvFbAeniHFka89/view?usp=sharing)
 and [stage2:appearance rendering](https://drive.google.com/file/d/1U8rp1hqOswQB-0T0fh2aDQu-o1GLfd6E/view?usp=sharing)


###  Acknowledgement

This repo is based on [RORD](https://github.com/Forty-lock/RORD), [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) and [Video-Depth-Anythingv2](https://github.com/DepthAnything/Video-Depth-Anything). Thanks for their wonderful works.


### Citation

```
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