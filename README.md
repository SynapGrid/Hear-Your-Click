# **Hear-Your-Click**  
**Interactive audio generation via object clicks**  

We present *Hear-Your-Click*, an interactive framework enabling targeted object audio generation via user clicks.

https://github.com/user-attachments/assets/2ca49ab5-80ca-42c4-b9a5-9dc7959ac358


> ⚠️ *Our codebase and model checkpoints are currently being uploaded and refined. We're prioritizing environment configuration and deployment guidelines to ensure consistent inference performance.*  



## 📦 **Installation**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/SynapGrid/Hear-Your-Click-2024.git 
   cd Hear-Your-Click-2024
   ```  

2. (Optional) Create a Conda environment:  
   ```bash  
   conda env create -n hyc python=3.9.11
   conda activate hyc
   ```

3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```


## 🚀 **Model Checkpoints**  
1. Download the model weights and place them in `./hyc_inference/inference/ckpt/`:  

- [epoch=000059.ckpt](https://drive.google.com/file/d/1QX24gEmN-cG03NlO0zT1geK1eUgOqDtk/view?usp=drive_link)  
- [epoch_10.pt](https://drive.google.com/file/d/15tbqXR-99QNg-Il6wxPD66q4EM4UkVvJ/view?usp=drive_link)
- [eval_classifier.ckpt](https://huggingface.co/SimianLuo/Diff-Foley/resolve/main/diff_foley_ckpt/eval_classifier.ckpt)
- [double_guidance_classifier.ckpt](https://huggingface.co/SimianLuo/Diff-Foley/resolve/main/diff_foley_ckpt/double_guidance_classifier.ckpt)

> 💡 Tips: you can use `gdown` and `wget` to download files. For example:
> ```bash  
> pip install gdown
> 
> cd ./hyc_inference/inference/ckpt
> 
> gdown https://drive.google.com/uc?id=1QX24gEmN-cG03NlO0zT1geK1eUgOqDtk 
> 
> gdown https://drive.google.com/uc?id=15tbqXR-99QNg-Il6wxPD66q4EM4UkVvJ
> 
> wget https://huggingface.co/SimianLuo/Diff-Foley/resolve/main/diff_foley_ckpt/eval_classifier.ckpt
> 
> wget https://huggingface.co/SimianLuo/Diff-Foley/resolve/main/diff_foley_ckpt/double_guidance_classifier.ckpt
> ```  

2. Download the models weights and place them in `./checkpoints`:

- [clap_clip.pt](https://github.com/MCR-PEFT/C-MCR/blob/main/checkpoints/clap_clip.pt)
- [laion_clap_fullset_fusion.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-fusion-best.pt)
- [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)



## 🧪 **Inference Command**  
Launch the inference demo:  
```bash  
python app.py --device cuda:0,1 --sam_model_type vit_b
```




