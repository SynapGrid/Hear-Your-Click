# **Hear-Your-Click**  
*arXiv: http://arxiv.org/abs/2507.04959*

We present *Hear-Your-Click*, an interactive framework enabling targeted object audio generation via user clicks.

https://github.com/user-attachments/assets/2ca49ab5-80ca-42c4-b9a5-9dc7959ac358



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

## 📚 **Citations**  
If you find this work useful for your research or applications, please cite our work:

**BibTeX**  
```bibtex
@misc{liang2025hearyourclickinteractivevideotoaudiogeneration,
      title={Hear-Your-Click: Interactive Video-to-Audio Generation via Object-aware Contrastive Audio-Visual Fine-tuning}, 
      author={Yingshan Liang and Keyu Fan and Zhicheng Du and Yiran Wang and Qingyang Shi and Xinyu Zhang and Jiasheng Lu and Peiwu Qin},
      year={2025},
      eprint={2507.04959},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.04959}, 
}
```

This project builds upon or references the following works. 

```bibtex
@misc{luo2023difffoleysynchronizedvideotoaudiosynthesis,
      title={Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models}, 
      author={Simian Luo and Chuanhao Yan and Chenxu Hu and Hang Zhao},
      year={2023},
      eprint={2306.17203},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2306.17203}, 
}

@misc{yang2023track,
      title={Track Anything: Segment Anything Meets Videos}, 
      author={Jinyu Yang and Mingqi Gao and Zhe Li and Shang Gao and Fangjing Wang and Feng Zheng},
      year={2023},
      eprint={2304.11968},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{wang2023connecting,
      title={Connecting Multi-modal Contrastive Representations}, 
      author={Zehan Wang and Yang Zhao and Xize Cheng and Haifeng Huang and Jiageng Liu and Li Tang and Linjun Li and Yongqi Wang and Aoxiong Yin and Ziang Zhang and Zhou Zhao},
      year={2023},
      eprint={2305.14381},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{sheffer2022i,
    title={I Hear Your True Colors: Image Guided Audio Generation},
    author={Roy Sheffer and Yossi Adi},
    year={2022},
    eprint={2211.03089},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```



