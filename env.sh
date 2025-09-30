# source /opt/conda/bin/activate 
conda env create -f environment.yml
conda activate flux_env
pip install wandb
pip install transformers==4.48.3
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121     --index-url https://download.pytorch.org/whl/cu121
pip install pandas
pip install -r diffusers/examples/advanced_diffusion_training/requirements_flux.txt
python -m pip install opencv-python-headless numpy
pip install numpy==1.26.4
pip install -e ./diffusers