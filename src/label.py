'''
label.py
Description: This module provides functionality for video segmentation using the SAM-2 model.
Authors: Matthew Hake, Ben Chidley, Garret Keyhani, Joshua Smith 
Date: 11/25/25
'''

#Source on how to impliment SAM2 for Video Segmentation: https://blog.roboflow.com/sam-2-video-segmentation/
import torch
from sam2.build_sam import build_sam2_video_predictor

CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"

sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)