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
   conda env create -n hyc python=3.9
   conda activate hyc
   ```

3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```


## 🚀 **Model Checkpoints**  
Download the model weights and place them in `./hyc_inference/inference/ckpt/`:  

- [Model 1](https://drive.google.com/file/d/1QX24gEmN-cG03NlO0zT1geK1eUgOqDtk/view?usp=drive_link)  
- [Model 2](https://drive.google.com/file/d/15tbqXR-99QNg-Il6wxPD66q4EM4UkVvJ/view?usp=drive_link)  

> 💡 Tips: use `gdown` to download Google Drive files:
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



## 🧪 **Inference Command**  
Launch the inference demo:  
```bash  
python app.py --device cuda:0,1 --sam_model_type vit_b
```


## 📚 **Citations**  
If you use this codebase or model checkpoints in your research, please cite our work:  

**BibTeX**  
```bibtex
@article{hear_your_click_2025,
  title = {Hear-Your-Click: Interactive Audio Generation via Object Clicks},
  author = {Your Name and Collaborators},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year = {2025},
  note = {Code available at \url{https://github.com/your-username/hear-your-click}}
}
```

This project builds upon or references the following works. 

```bibtex
@article{hear_your_click_2025,
  title = {Hear-Your-Click: Interactive Audio Generation via Object Clicks},
  author = {Your Name and Collaborators},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year = {2025},
  note = {Code available at \url{https://github.com/your-username/hear-your-click}}
}
```



