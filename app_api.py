import random
import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import sys
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch 
from tools.painter import mask_painter
import psutil
import time


import librosa

try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")

import os
from transformers import AutoModel, AutoTokenizer
import shutil
import subprocess
import pickle

from cmcr.cmcr_model_modified import C_MCR_CLAPCLIP
from cmcr.cmcr_model_modified import ModalityType

import torch.nn.functional as F

# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    # 保存到地址
    print("保存视频")
    if os.path.exists('./result/tmp'):
        try:
            shutil.rmtree('./result/tmp')
        except OSError as e:
            print("e")
    os.makedirs('./result/tmp',exist_ok=True)

    # 裁剪视频
    print("裁剪视频")
    new_video_input = os.path.join('./result/tmp',os.path.basename(video_input))
    cmd = ["ffmpeg", "-ss", "00:00:00","-to", "00:00:08.12", "-y","-i",video_input, "-c","copy",new_video_input]
    subprocess.check_call(cmd)

    video_path = new_video_input

    print
    # video_fps4 = os.path.join('./result/video_fps4',os.path.basename(video_input))
    frames = []
    user_name = time.time()
    operation_log = [("",""),("Upload video already.","Normal")]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1])

    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps,
        'origin_path': video_path
        }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"], video_state["fps"], len(frames), image_size)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_info, video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True, value=operation_log)

def run_example(example):
    return video_input
# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    # update the masks when select a new template frame
    if video_state["masks"][image_selection_slider] is not None:
        video_state["painted_images"][image_selection_slider] = mask_painter(video_state["origin_images"][image_selection_slider], video_state["masks"][image_selection_slider])
    if mask_dropdown:
        print("ok")
    operation_log = [("",""), ("Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider),"Normal")]


    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Set the tracking finish at frame {}".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log

def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("",""), (" ","Normal")]
    return painted_image, video_state, interactive_state, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("",""),(" ","Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("","")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Clear points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all mask, please add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("")]
    return select_frame, operation_log


def calculate_rms(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算均方根值
    rms = np.sqrt(np.mean(gray.astype(float) ** 2))
    return rms

def df_inference(video_path,video_output,extract_cavp,latent_diffusion_model,save_path):
    seed_everything(21) #21
    import subprocess
    # 在这里降fps
    print("preprocessing...")
    if os.path.exists('./result/video_fps4'):
        try:
            shutil.rmtree('./result/video_fps4')
        except OSError as e:
            print("e")
    os.makedirs('./result/video_fps4',exist_ok=True)
    cmd = ["ffmpeg","-y","-i",video_path,"-filter:v","fps=4",os.path.join('./result/video_fps4',os.path.basename(video_path))]
    subprocess.check_call(cmd)

    # 在这里提取clip_feature
    print("clip features...")
    if os.path.exists('./result/clip_pickle'):
        try:
            shutil.rmtree('./result/clip_pickle')
        except OSError as e:
            print("e")
    os.makedirs('./result/clip_pickle',exist_ok=True)
    cmd = [
        "python", 
        "im2wav/Data/preprocess/collect_video_CLIP.py",
        "-videos_dir", "./result/video_fps4",
        "-save_dir", "./result/clip_pickle",
        "-bs", "2"
    ]
    try:
        # 运行命令
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Command executed successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}:")
        print(e.stderr)

    # 转cavp格式
    os.makedirs('./result/clip_cavp',exist_ok=True)
    for name in os.listdir('./result/clip_pickle'):
        name = name[:-7]
        pklpath = os.path.join('./result/clip_pickle',name+".pickle")
        with open(pklpath,'rb') as file:  # 以二进制读模式（rb）打开pkl文件
            data = pickle.load(file)
            print(data[name].shape)
            np.save(os.path.join("./result/clip_cavp",name+'.npy'),np.asarray(data[name]))


    # Extract Video CAVP Features & New Video Path:
    cavp_feats = extract_cavp(os.path.join('./result/video_fps4',os.path.basename(video_path)),video_path, start_second, truncate_second, tmp_path=tmp_path)
    print("cavp_feats",cavp_feats.shape)
    clip = np.load(os.path.join('./result/clip_cavp',os.path.basename(video_path)[:-4]+'.npy'),allow_pickle=True)
    print("clipfea:",clip.shape)
    cavp_feats = clip[:cavp_feats.shape[0],:]+cavp_feats
    cavp_feats = F.normalize(torch.tensor(cavp_feats), dim=-1)
    cavp_feats = np.asarray(cavp_feats)
    print("cavp_clip_feats",cavp_feats.shape)

    ################# Diff-foley Generation #################
    # Whether use Double Guidance:
    use_double_guidance = True

    if use_double_guidance:
        classifier_config_path = "./diff_foley_inference/inference/config/Double_Guidance_Classifier.yaml"
        classifier_ckpt_path = "./diff_foley_inference/inference/diff_foley_ckpt/double_guidance_classifier.ckpt"
        classifier_config = OmegaConf.load(classifier_config_path)
        classifier = load_model_from_config(classifier_config, classifier_ckpt_path)

    sample_num = 10 # sample_num

    # Inference Param:
    cfg_scale = 4.5      # Classifier-Free Guidance Scale
    cg_scale = 50        # Classifier Guidance Scale
    steps = 50                # Inference Steps
    sampler = "DPM_Solver"    # DPM-Solver Sampler


    # save_path = save_path + "_CFG{}_CG{}_{}_{}_useDG_{}".format(cfg_scale, cg_scale, sampler, steps, use_double_guidance)
    if os.path.exists(save_path):
        try:
            shutil.rmtree(save_path)
        except OSError as e:
            print("e")
    os.makedirs(save_path, exist_ok=True)

    # Video CAVP Features:
    video_feat = torch.from_numpy(cavp_feats).unsqueeze(0).repeat(sample_num, 1, 1).to(device)
    print(video_feat.shape)

    # Truncate the Video Cond:
    feat_len = video_feat.shape[1]
    print("feat_len:",feat_len)
    truncate_len = 32 #32
    window_num = feat_len // truncate_len
    print("window_num:",window_num)
    if window_num == 0:
            window_num=1

    audio_list = []     # [sample_list1, sample_list2, sample_list3 ....]
    for i in tqdm(range(window_num), desc="Window:"):
        start, end = i * truncate_len, (i+1) * truncate_len
        
        # 1). Get Video Condition Embed:
        embed_cond_feat = latent_diffusion_model.get_learned_conditioning(video_feat[:, start:end])     

        # 2). CFG unconditional Embedding:
        uncond_cond = torch.zeros(embed_cond_feat.shape).to(device)
        
        # 3). Diffusion Sampling:
        print("Using Double Guidance: {}".format(use_double_guidance))
        if use_double_guidance:
            audio_samples, _ = latent_diffusion_model.sample_log_with_classifier_diff_sampler(embed_cond_feat, origin_cond=video_feat, batch_size=video_feat.shape[0], sampler_name=sampler, ddim_steps=steps, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=uncond_cond,classifier=classifier, classifier_guide_scale=cg_scale)  # Double Guidance
        else:
            audio_samples, _ = latent_diffusion_model.sample_log_diff_sampler(embed_cond_feat, batch_size=sample_num, sampler_name=sampler, ddim_steps=steps, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=uncond_cond)           #  Classifier-Free Guidance

        # 4). Decode Latent:
        audio_samples = latent_diffusion_model.decode_first_stage(audio_samples)                     
        audio_samples = audio_samples[:, 0, :, :].detach().cpu().numpy()                               

        # 5). Spectrogram -> Audio:  (Griffin-Lim Algorithm)
        sample_list = []        #    [sample1, sample2, ....]
        for k in tqdm(range(audio_samples.shape[0]), desc="current samples:"):
            sample = inverse_op(audio_samples[k])
            sample_list.append(sample)
        audio_list.append(sample_list)
    
    # Save Samples:
    path_list = []
    for i in range(sample_num):      # sample_num
        current_audio_list = []
        for k in range(window_num):
            current_audio_list.append(audio_list[k][i])
        current_audio = np.concatenate(current_audio_list,0)
        print(current_audio.shape)
        sf.write(os.path.join(save_path, "sample_{}_diff.wav").format(i), current_audio, 16000)
        path_list.append(os.path.join(save_path, "sample_{}_diff.wav").format(i))
    

    
    # Concat The Video and Sound:
    import subprocess

    # Cut video
    cmd = ["ffmpeg", "-ss", "00:00:00","-to", "00:00:08.12", "-y","-i",video_output, "-c","copy",video_output[:-4]+"_cut.mp4"]
    subprocess.check_call(cmd)

    # silent video
    def which_ffmpeg() -> str:
        result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
        return ffmpeg_path

    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
    cmd += f'-y -i {video_output[:-4]+"_cut.mp4"} -an -filter:v fps=fps={21} {video_output[:-4]+"_cut.mp4"}'
    subprocess.call(cmd.split())

    # merge video and audio
    for i in range(sample_num):
        gen_audio_path = path_list[i]
        out_path = os.path.join(save_path, "output_{}.mp4".format(i))
        cmd = ["ffmpeg","-y","-i",video_output[:-4]+"_cut.mp4","-i",gen_audio_path,"-c:v","copy","-c:a","aac","-map","0:v:0","-map","1:a:0","-shortest",out_path]
        subprocess.check_call(cmd)
    print("Gen Success !!")

    

    # cmcr evaluation
    score1 = []
    for i in range(sample_num):
        # 对一个样本进行推理，CMCR-VA = AVG(CMCR-Image-Audio)
        wav_path = os.path.join(save_path,f"sample_{i}_diff.wav")
        
        input = {ModalityType.VISION: video_path,
                 ModalityType.AUDIO: wav_path}

        # you can get single modality embeddings by using these functions
        v_emb = clap_clip_model.get_vision_embedding(input,frame_num=10)
        a_emb = clap_clip_model.get_audio_embedding(input,frame_num=10)
        v_emb = torch.mean(v_emb,dim=0)
        a_emb = torch.mean(a_emb,dim=0)

        similarity =  a_emb @ v_emb.T * 10.0
        if similarity>0:
            score1.append(i)
    
    if len(score1) <=0:
        score1 = range(sample_num)

    # score2 = []
    # 视频文件路径
    eval_video_path = video_path
    cap = cv2.VideoCapture(eval_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 存储RMS值
    rms_values_v = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有更多帧，则退出循环
        # 计算当前帧的RMS值
        rms = calculate_rms(frame)
        rms_values_v.append(rms)

    cap.release()

    best_score=-100
    for i in score1:
        # 加载音频文件
        audio_path = os.path.join(save_path,f'sample_{i}_diff.wav')
        y, sr = librosa.load(audio_path, sr=None)  # sr=None表示使用原始采样率

        # 计算帧长和帧移
        frame_length = 2048  # 每个分析窗口的长度
        hop_length = 512    # 相邻窗口之间的跳跃步长

        # 计算RMS能量
        rms_values_a = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

        # 将RMS值转换为一维数组
        rms_values_a = rms_values_a[0]

        # 创建时间轴
        frames = range(len(rms_values_a))
        time = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        # 对齐长度
        min_length = min(len(rms_values_v), len(rms_values_a))
        rms_values_v = rms_values_v[:min_length]
        rms_values_a = rms_values_a[:min_length]

        # 计算皮尔逊相关系数
        pearson_corr = np.corrcoef(rms_values_v, rms_values_a)[0, 1]
        if pearson_corr > best_score:
            best_idx=i

    # 对每个样本的两个分数进行排序（降序
    # sort1 = torch.argsort(torch.tensor(score1),dim=0,descending=True)
    # sort2 = torch.argsort(torch.tensor(score2),dim=0,descending=True)
    # print(sort1)
    # sort = sort1+sort2
    # best_idx = torch.argmin(sort)


    # output the final video
    generated_video_audio_path = os.path.join(save_path, f"output_{best_idx}.mp4")

    return generated_video_audio_path


# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    start_time=time.time()
    operation_log = [("",""), ("Track the selected masks, and then you can select the masks for inpainting.","Normal")]
    model.xmem.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.","Error"), ("","")]
        # return video_output, video_state, interactive_state, operation_error
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask) # 生成mask
    # clear GPU memory
    model.xmem.clear_memory()

    if interactive_state["track_end_number"]: 
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    video_output = generate_video_from_frames(video_state["painted_images"], output_path="./result/track/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video
    
    
    interactive_state["inference_times"] += 1
    
    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(interactive_state["inference_times"], 
                                                                                                                        interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
                                                                                                                        interactive_state["positive_click_times"],
                                                                                                                        interactive_state["negative_click_times"]))

    #### shanggao code for mask save
    interactive_state["mask_save"]=True
    if interactive_state["mask_save"]:
        if not os.path.exists('./result/mask/{}'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/mask/{}'.format(video_state["video_name"].split('.')[0]))
        i = 0
        print("save mask")
        for mask in video_state["masks"]:
            np.save(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.npy'.format(i)), mask)
            i+=1

    print("Generating Model Input...")
    masked_frame = []
    for idx,mask in enumerate(video_state["masks"]):
        processed = video_state["origin_images"][video_state["select_frame_number"]+idx].copy()
        for k in range(3):         
            processed[:,:,k] = processed[:,:,k] * mask
        masked_frame.append(processed)
    black_mask_video =  generate_video_from_frames(masked_frame, output_path="./result/model_input/{}".format(video_state["video_name"]), fps=fps)


    # black_mask_video = video_state['origin_video_path']
    # 保存：
    print("black_mask_video:", black_mask_video)
    
    
    print("Generation Finished.")


    print("Generating Audio...")
    end_time = time.time()
    print(f"视频追踪分割运行时间：{end_time-start_time}s")
    # generated_video_audio_path = df_inference("./result/model_input/{}".format(video_state["video_name"]))
    start_time=time.time()
    generated_video_audio_path = df_inference(black_mask_video,video_output,extract_cavp,latent_diffusion_model,save_path)

    
    end_time = time.time()
    print(f"音效生成时间：{end_time-start_time}s")
       
        # save_mask(video_state["masks"], video_state["video_name"])
    #### shanggao code for mask save
    
    return generated_video_audio_path, video_state, interactive_state, operation_log

# extracting masks from mask_dropdown
# def extract_sole_mask(video_state, mask_dropdown):
#     combined_masks = 
#     unique_masks = np.unique(combined_masks)
#     return 0 

# inpaint 
def inpaint_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Removed the selected masks.","Normal")]

    frames = np.asarray(video_state["origin_images"])
    fps = video_state["fps"]
    inpaint_masks = np.asarray(video_state["masks"])
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    # convert mask_dropdown to mask numbers
    inpaint_mask_numbers = [int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))]
    # interate through all masks and remove the masks that are not in mask_dropdown
    unique_masks = np.unique(inpaint_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in inpaint_mask_numbers:
            continue
        inpaint_masks[inpaint_masks==i] = 0
    # inpaint for videos

    try:
        inpainted_frames = model.baseinpainter.inpaint(frames, inpaint_masks, ratio=interactive_state["resize_ratio"])   # numpy array, T, H, W, 3
    except:
        operation_log = [("Error! You are trying to inpaint without masks input. Please track the selected mask first, and then press inpaint. If VRAM exceeded, please use the resize ratio to scaling down the image size.","Error"), ("","")]
        inpainted_frames = video_state["origin_images"]
    video_output = generate_video_from_frames(inpainted_frames, output_path="./result/inpaint/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video

    return video_output, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    # height, width, layers = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # print(output_path)
    # for frame in frames:
    #     video.write(frame)
    
    # video.release()
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=int(fps), video_codec="libx264")
    return output_path


# args, defined in track_anything.py
args = parse_augment()

# check and download checkpoints if needed
SAM_checkpoint_dict = {
    'vit_h': "sam_vit_h_4b8939.pth",
    'vit_l': "sam_vit_l_0b3195.pth", 
    "vit_b": "sam_vit_b_01ec64.pth"
}
SAM_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


folder ="./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
args.port = 12212
args.device = "cuda:0"
# args.mask_save = True

# initialize sam, xmem, e2fgvi models
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)


title = """<p><h1 align="center">Hear Your Click</h1></p>
    """
description = """<p align="center">Upload your video and click on the frame to initiate object tracking and audio generation. You can click multiple times to refine the tracked area.</p>"""


from diff_foley_inference.inference.demo_util import Extract_CAVP_Features,inverse_op
from diff_foley_inference.diff_foley.util import instantiate_from_config
from omegaconf import OmegaConf
from tqdm import tqdm
import soundfile as sf

fps = 4                                                     #  CAVP default FPS=4, Don't change it.
batch_size = 40   # Don't change it.
cavp_config_path = "./diff_foley_inference/inference/config/Stage1_CAVP.yaml"              #  CAVP Config # 最好cd进diff-foley根目录执行，或者写绝对路径，否则会报错
cavp_ckpt_path = "./diff_foley_inference/inference/diff_foley_ckpt/epoch_10.pt"      #  CAVP Ckpt

################# Load models #################
device = torch.device("cuda")

# Initalize CAVP Model:
extract_cavp = Extract_CAVP_Features(fps=fps, batch_size=batch_size, device=device, config_path=cavp_config_path, ckpt_path=cavp_ckpt_path)

clap_clip_model = C_MCR_CLAPCLIP(device="cpu")



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu",weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(21) #21


# LDM Config:
ldm_config_path = "./diff_foley_inference/inference/config/Stage2_LDM.yaml"
ldm_ckpt_path = "./diff_foley_inference/inference/diff_foley_ckpt/epoch=000059.ckpt"
# "./inference/diff_foley_ckpt/ldm_epoch240.ckpt" 
config = OmegaConf.load(ldm_config_path)

# Loading LDM:
latent_diffusion_model = load_model_from_config(config, ldm_ckpt_path)


tmp_path = "./result/temp_folder" 
save_path = "./result/df_output"
start_second = 0              # Video start second
truncate_second = 8.2         # Video end = start_second + truncate_second


with gr.Blocks() as iface:
    """
        state for 
    """
    click_state = gr.State([[],[]])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        "resize_ratio": 1
    }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        }
    )
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():

        # for user video input
        with gr.Column():
            with gr.Row(scale=0.4):
                video_input = gr.Video(autosize=True)
                with gr.Column():
                    video_info = gr.Textbox(label="Video Info")
                    # resize_info = gr.Textbox(value=" ")
                    resize_ratio_slider = gr.Slider(minimum=0.02, maximum=1, step=0.02, value=1, label="Resize ratio", visible=True)
          

            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 

                     # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive",  "Negative"],
                                value="Positive",
                                label="Point prompt",
                                interactive=True,
                                visible=False)
                            remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False) # Remove mask   Synthesis mask
                            clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False).style(height=160) # Clear clicks  Seperate mask
                            Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False) # Add Mask  Repair mask
                    template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False).style(height=360)
                    image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="start frame", visible=False)
                    track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="end frame", visible=False)
            
                with gr.Column():
                    run_status = gr.HighlightedText(value=[("Text","Error"),("to be","Label 2"),("highlighted","Label 3")], visible=False)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                    video_output = gr.Video(autosize=True, visible=False).style(height=360)
                    with gr.Row():
                        tracking_video_predict_button = gr.Button(value="Generate Audio", visible=False) # Tracking
                        # inpaint_video_predict_button = gr.Button(value="Auto", visible=False) # Inpainting

    # first step: get the video information 
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, video_state
        ],
        outputs=[video_state, video_info, template_frame,
                 image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, Add_mask_button, template_frame,
                 tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, run_status]
    )   

    # second step: select images from slider
    image_selection_slider.release(fn=select_template, 
                                   inputs=[image_selection_slider, video_state, interactive_state], 
                                   outputs=[template_frame, video_state, interactive_state, run_status], api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number, 
                                   inputs=[track_pause_number_slider, video_state, interactive_state], 
                                   outputs=[template_frame, interactive_state, run_status], api_name="end_image")
    resize_ratio_slider.release(fn=get_resize_ratio, 
                                   inputs=[resize_ratio_slider, interactive_state], 
                                   outputs=[interactive_state], api_name="resize_ratio")
    
    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status]
    )

    # inpaint video from select image and mask
    # inpaint_video_predict_button.click(
    #     fn=inpaint_video,
    #     inputs=[video_state, interactive_state, mask_dropdown],
    #     outputs=[video_output, run_status]
    # )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status]
    )
    
    # clear input
    video_input.clear(
        lambda: (
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        },
        {
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": 0,
        "resize_ratio": 1
        },
        [[],[]],
        None,
        None,
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=[]), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False)
                        
        ),
        [],
        [ 
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            tracking_video_predict_button, image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
            Add_mask_button, template_frame, tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, run_status
        ],
        queue=False,
        show_progress=False)

    # points clear
    clear_button_click.click(
        fn = clear_click,
        inputs = [video_state, click_state,],
        outputs = [template_frame,click_state, run_status],
    )
    # set example
    gr.Markdown("##  Examples")
    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "./test_sample/", test_sample) for test_sample in os.listdir(os.path.join(os.path.dirname(__file__), "./test_sample/"))],
        fn=run_example,
        inputs=[
            video_input
        ],
        outputs=[video_input],
        # cache_examples=True,
    )


import uvicorn
from fastapi import FastAPI
HOST = os.getenv("ARNOLD_WORKER_0_HOST")
PORT = int(os.getenv("ARNOLD_WORKER_0_PORT").split(",")[-1])
print("==> HOST: {}".format(HOST))
print("==> PORT: {}".format(PORT))
print("host: {}, port: {}".format(HOST, PORT))
print("https://data-edu.bytedance.net/proxy/gradio/host/[{}]:{}/".format(
    HOST, PORT
))
fapp = FastAPI()
iface.queue(concurrency_count=1)
app = gr.mount_gradio_app(fapp, iface, path='/proxy/gradio/host/[{}]:{}'.format(HOST, PORT))
# iface.launch(debug=True, enable_queue=True, server_port=args.port, server_name="0.0.0.0",share=True)
uvicorn.run(app,
            port=PORT, 
            host='0000:0000:0000:0000:0000:0000:0000:0000',
            timeout_keep_alive=60)
# iface.launch(debug=True, enable_queue=True)