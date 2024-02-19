import argparse
import glob
import os
import json
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import pycocotools.mask as maskUtils
from marinegpt.common.config import Config
from marinegpt.common.dist_utils import get_rank
from marinegpt.common.registry import registry
from marinegpt.conversation.conversation import Chat, CONV_VISION
from PIL import Image
# imports modules for registration
from marinegpt.datasets.builders import *
from marinegpt.models import *
from marinegpt.processors import *
from marinegpt.runners import *
from marinegpt.tasks import *

# import spacy
# nlp = spacy.load('en_core_web_sm')
# def get_noun_phrases(text):
#     doc = nlp(text)
#     noun_phrases = []
#     for chunk in doc.noun_chunks:
#         noun_phrases.append(chunk.text)
#     return noun_phrases

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--img_path", required=True, help="path of images")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

for files in glob.glob(os.path.join(args.img_path,"*.jpg")):
    chat_state = CONV_VISION.copy()
    chat_state.messages = [['Human', '<Img><ImageHere></Img> Describe the object in this figure']]
    img_list = []
    if os.path.exists(files.replace(".jpg",".txt")):
        continue
    write_data=open(files.replace(".jpg",".txt"),"w")
    patch_large=Image.open(files).convert('RGB')
    captions=chat.generate_caption(patch_large,chat_state,img_list)
    write_data.write(captions)
    write_data.close()
    print(captions)
