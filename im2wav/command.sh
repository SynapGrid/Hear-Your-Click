
# 提取clip特征 进入run文件夹运行

python ../Data/preprocess/collect_video_CLIP.py 
-videos_dir /home/ma-user/work/project/im2wav-main/demovideos/fps4 
-save_dir /home/ma-user/work/project/im2wav-main/demovideos/rvs_clip_fea 
-bs 100 


# 推理

python ../models/sample.py 
-bs 2 
-experiment_name video_CLIP 
-CLIP_dir /data/audiodataset/l50039443/other_inference/im2wav/maskedvideo_test_fps4_clip_fea_len40 
-models im2wav 
-save_dir /data/audiodataset/l50039443/other_inference/im2wav/maskedvideo_test_im2wav_results


python ../models/sample.py 
-bs 2 
-experiment_name video_CLIP 
-CLIP_dir /data/audiodataset/l50039443/other_inference/im2wav/masked_video_test_originfps/len122_pickle 
-models im2wav 
-save_dir /data/audiodataset/l50039443/other_inference/im2wav/masked_video_test_originfps/maskedvideo_results_originfps_len122
