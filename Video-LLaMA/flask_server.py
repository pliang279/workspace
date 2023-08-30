from flask import Flask, request
import os
from functools import cache

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat
import decord
decord.bridge.set_bridge('torch')

from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

from predict import parse_args, get_description

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'flv', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@cache
def load_model():
    print('Initializing Chat')
    
    args = parse_args()
    print(args)
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')
    return chat

@app.before_first_request
def before_first_request():
    initialize_app(app)

def initialize_app(app):
    app.config['MODEL'] = load_model()
    print("model loaded")
    app.config['PROMPT'] = "Describe the video briefly"
    app.config['NUM_BEAMS'] = 1
    app.config['TEMPERATURE'] = 1.0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No file part'
        
        video = request.files['video']
        
        if video.filename == '':
            return 'No selected file'
        if video and allowed_file(video.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(filename)
            print("saved video")
            result = get_description(filename, app.config['MODEL'], app.config['PROMPT'], app.config['NUM_BEAMS'], app.config['TEMPERATURE'])
            return result
            

    return '''
    <!doctype html>
    <title>Upload Video</title>
    <h1>Upload Video</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=video>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    load_model()
    app.run(port=5000)