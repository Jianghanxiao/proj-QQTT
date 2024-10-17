import os
import glob

videos_path = (
    "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/reform_SG/burger/videos"
)


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    videos = glob.glob(f"{videos_path}/*.mp4")
    exist_dir(f"{videos_path}/../depths")
    for video in videos:
        os.system(
            f"python ../DepthCrafter/run.py --video-path {video} --target-fps -1 --save-folder {videos_path}/../depths_depthcrafter"
        )
