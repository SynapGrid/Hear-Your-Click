
# 引用库：track anything, c-mcr, im2wav, diff-foley, 


# 
conda create -n hyc python=3.9
conda activate hyc

pip install -r requirements_origin.txt
cd im2wav

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nsight-compute-2024.1.1/host/linux-desktop-glibc_2_11_3-x64/Mesa:$LD_LIBRARY_PATH 
python app.py --device cuda:0
python app_api.py --device cuda:0,1 --sam_model_type vit_b

opencv-python==4.6.0.66
opencv-python-headless==4.7.0.68
conda install -c conda-forge ffmpeg
conda install -c conda-forge opencv

# I uninstalled opencv from my conda environment ( conda uninstall opencv)

# Installed latest ffmpeg using conda-forge channel

# Then installed opencv again using conda-forge channel ( conda install -c conda-forge opencv) . Don't install using menpo channel







