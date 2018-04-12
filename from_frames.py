from moviepy.editor import ImageSequenceClip
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create video.')
    parser.add_argument(
        '--video',
        type=str,
        default='result',
        help='Video file name.'
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        default='Results',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='FPS (Frames per second) setting for the video.')

    args = parser.parse_args()

    video_file = args.video + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
