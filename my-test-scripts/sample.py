import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file="configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py"
checkpoint_file="checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"

device = torch.device("cuda:0")

model = init_recognizer(config_file, checkpoint_file, device=device)

video = "demo/demo.mp4"
labels = "tools/data/kinetics/label_map_k400.txt"
results = inference_recognizer(model, video)

labels = open("tools/data/kinetics/label_map_k400.txt").readlines()