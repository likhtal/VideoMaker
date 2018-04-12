import argparse
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def main():
    global image_folder

    parser = argparse.ArgumentParser(description='Create video.')
    parser.add_argument(
        'video',
        type=str,
        help='Video file name.'
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        default='Heat',
        help='Path to image folder. The video will be split into frames there.'
    )

    args = parser.parse_args()
    image_folder = args.image_folder 

    clip = VideoFileClip(args.video)
    clip.write_images_sequence(args.image_folder + "/frame%04d.jpeg")

if __name__ == '__main__':
    main()
