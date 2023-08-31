import subprocess
import argparse
 
def detect(filename):
    subprocess.run(['yolo','task=detect', 'mode=predict', 'model=yolov8x.pt', 'source='+filename], check=True, capture_output=True).stdout

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection.')
    parser.add_argument('filename')
    args = parser.parse_args()
    detect(args.filename)