import subprocess
import argparse
 
def detect(filename):
    subprocess.run(['yolo','task=segment', 'mode=predict', 'model=yolov8x-seg.pt', 'conf=0.5', 'source='+filename], check=True, capture_output=True).stdout

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Segmentation.')
    parser.add_argument('filename')
    args = parser.parse_args()
    detect(args.filename)