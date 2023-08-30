import ffmpeg, os
import datetime 
import time
import json
import cv2
from time import mktime


def get_secs_hms(x):
    x = time.strptime(x.strip(),'%H:%M:%S')
    x = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    return x


def get_secs_ms(x):
    x = time.strptime(x.strip(),'%M:%S')
    x = datetime.timedelta(minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    return x


def trim_cv2(input_path, info):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    section_idx = 0

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cur_output_prefix, cur_start_time, cur_end_time = info[section_idx]
        if cur_end_time == 'end':
            cur_end_time = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count += 1
    
        if frame_count >= cur_start_time:
            output_path = f"{cur_output_prefix}/clip.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            
            while frame_count <= cur_end_time:
                out.write(frame)
                ret, frame = cap.read()
                frame_count += 1
                
            out.release()
            section_idx += 1
    cap.release()




def trim_ffmpeg(input, info):
    if not os.path.exists(input):
        return 
    input_stream = ffmpeg.input(input, fflags="+discardcorrupt")
    pts = "PTS-STARTPTS"
    for i in info:
        dir, start, end= i
        if os.path.exists(f'{dir}/clip.mp4'):
            print(dir)
            continue
        if end == 'end':
            end = int(float(ffmpeg.probe(input)["format"]["duration"]))
        video = input_stream.trim(start=start, end=end).setpts(pts)
        audio = (input_stream
                .filter_("atrim", start=start, end=end)
                .filter_("asetpts", pts))
        video_and_audio = ffmpeg.concat(video, audio, v=1, a=1)
        output = ffmpeg.output(video_and_audio, f'{dir}/clip.mp4', format="mp4")
        output.run(overwrite_output=True)


if __name__ == "__main__":
    # dataset = 'Werewolf/youtube/data.json'
    # game_data = dict()
    # t = datetime.datetime(1900, 1, 1, 0, 0, 0)

    # with open(dataset) as file:
    #     datafile = json.load(file)
    # for data in datafile:
    #     file = data['video_name'] 
    #     info = (data['startTime'], data['endTime'], data['Game_ID'])
    #     if file not in game_data:
    #         game_data[file] = [info]
    #     else:
    #         game_data[file].append(info)

    # for k,v in game_data.items():
    #     if k == 'One Night Ultimate Werewolf 381 The Hunt Is On':
    #         continue
    #     v = sorted(v, key=lambda x: int(x[2][-1]))
    #     trim_ffmpeg(k, v)



    for subdir, dirs, files in os.walk('./Werewolf/youtube/transcripts'):
        for file in files:

            if file == '.DS_Store':
                continue

            filename = file.split('.')[0]
            video_path = './video_clip/'+filename+'.mp4'

            
            with open(subdir+'/'+file, 'r') as f:
                transcript = f.read().split('\n')
            
            info = []

            for i in range(len(transcript)-1):
                marker = str(i) + '_' + transcript[i].split(' ')[0]
                temp_dir = f'slices/{filename}/{marker}'
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                start_time = transcript[i][transcript[i].index('(')+1: transcript[i].index(')')]
                start_time = get_secs_ms(start_time)

                if i == len(transcript)-2:
                    end_time = 'end'

                else:
                    end_time = transcript[i+1][transcript[i+1].index('(')+1: transcript[i+1].index(')')]
                    if end_time == start_time:
                        end_time += 1
                    end_time = get_secs_ms(end_time)
                with open(temp_dir+'/script.txt', 'w') as text:
                    text.write(transcript[i])
                info.append([temp_dir,start_time,end_time])
            
            trim_ffmpeg(video_path, info)