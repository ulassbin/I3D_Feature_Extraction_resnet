from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
from resnet import i3_res50
import os
from moviepy.editor import VideoFileClip
from PIL import Image  # Import Image module from PIL
import cv2


def extract_all_frames_with_padding(input_video, output_path, temporal_padding="same", padding_size=0):
    """
    Extracts all frames from a video with padding at the start and end, and saves them as images.
    
    Args:
        input_video (str): Path to the input video file.
        output_path (str): Path to save the output frames.
        padding_size (int): Size of padding in seconds.
        temporal_padding (str): Type of temporal padding ("same" for cloning first and last frames, "zeros" for zero padding).
        
    Returns:
        None
    """
    # Load the video clip
    clip = VideoFileClip(input_video)
    
    vid_frames = get_video_length_raw(input_video)#int(clip.duration * clip.fps)
    # Get total number of frames in the video
    total_frames = vid_frames + 2 * padding_size
    
    # Print total number of frames
    print("Total number of frames {} (including padding {}):".format(vid_frames,total_frames))
    
    # Get first and last frames for temporal padding

    
    # Loop through all frames in the video (Can override here with opencv...)
    offset = padding_size
    for i, frame in enumerate(clip.iter_frames(fps=clip.fps)):
        # Save frame as an image
        frame_path = f"{output_path}/frame_{i+padding_size}.jpg"
        frame_image = Image.fromarray(frame)
        frame_image.save(frame_path)
        if(i == 0):
            start_padding_frame = frame_image
        if(i==vid_frames-1):
            end_padding_frame = frame_image

#    if temporal_padding == "same":
#        start_padding_frame = clip.get_frame(1)
#        end_padding_frame = clip.get_frame(clip.duration)
    if temporal_padding == "zeros":
        start_padding_frame = np.zeros_like(start_padding_frame)
        end_padding_frame = np.zeros_like(end_padding_frame)

    
    # Add start padding frames
    for i in range(padding_size):
        frame_path = f"{output_path}/frame_{i}.jpg"
        start_padding_frame.save(frame_path)
    
    # Add end padding frames
    for i in range(vid_frames, total_frames):
        frame_path = f"{output_path}/frame_{i}.jpg"
        end_padding_frame.save(frame_path)


def get_video_length_raw(video_path): 
    # Just checks it by manually opening up frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()

    return frame_count

def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode, video_extension, temporal_padding="none"):
	Path(outputpath).mkdir(parents=True, exist_ok=True)
	temppath = outputpath+ "temp/"
	rootdir = Path(datasetpath)
	videos = [str(f) for f in rootdir.glob('**/*.{}'.format(video_extension))]
	# setup the model
	i3d = i3_res50(400, pretrainedpath)
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode
	for video in videos:
		print("{} with len {}".format(video, get_video_length_raw(video)))
		videoname = video.split("/")[-1].split(".")[0]
		startime = time.time()
		Path(temppath).mkdir(parents=True, exist_ok=True) # Clear Initially
		shutil.rmtree(temppath) # Clear initially
		Path(temppath).mkdir(parents=True, exist_ok=True) # Reopen
		extract_all_frames_with_padding(video, temppath, temporal_padding, 8)
		features = run(i3d, frequency, temppath, batch_size, sample_mode)
		print("Obtained features of size: ", features.shape)
		np.save(outputpath + "/" + videoname, features)
		del features
		torch.cuda.empty_cache()
		shutil.rmtree(temppath)
		print("done in {0}.".format(time.time() - startime))

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="samplevideos/")
	parser.add_argument('--outputpath', type=str, default="output")
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	parser.add_argument("--temporal_padding", type=str, default="zeroes")
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode, "" ,args.temporal_padding)    
