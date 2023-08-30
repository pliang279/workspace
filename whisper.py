import whisperx
import librosa
import os
import glob
import json
from tqdm import tqdm
import numpy as np

def generate_words(audio_file, model, model_a):
    device = "cuda" 
    batch_size = 16
    compute_type = "int8"

    try:
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        duration = librosa.get_duration(filename=audio_file)
        #if len(result['segments']) != 1:
        #    import pdb; pdb.set_trace()

        start = result["segments"][0]['start']
        end = result["segments"][-1]['end']

        texts = []
        words = []
        for segment in result["segments"]:
            texts.append(segment["text"])
            words.append(segment["words"])
    except:
        start = -1
        end = -1
        texts = []
        words = []
    try:
        data, rate = librosa.load(audio_file, res_type='kaiser_fast', duration=1000, sr=22050*2, offset=0)
        rms = librosa.feature.rms(y=data)[0]
        avg_rms = np.mean(rms)
        loudness_in_db = 20 * np.log10(avg_rms)
    except:
        loudness_in_db = -100
    return (start, end, texts, words, loudness_in_db)


# load many json file under one directory
def load(dir_path):
    wav_files = glob.glob(os.path.join(dir_path, '*.wav'))
    wav_files.sort()
    return wav_files

if __name__ == '__main__':
    #files = load('./data/ttm_audio_context')
    files = ['test.wav']
    device = "cuda" 
    batch_size = 16
    compute_type = "int8"
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language='en')
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    for idx, file in enumerate(tqdm(files)):
        start, end, texts, words, loudness_in_db = generate_words(file, model, model_a)
        wav_file_name = file.split('/')[-1].split('.')[0]
        with open(f'./data/ttm_words_context/{wav_file_name}.json', 'w') as f:
            json.dump({'start': start, 'end': end, 'texts': texts, 'context_words': words, 'loudness_in_db': loudness_in_db}, f)
