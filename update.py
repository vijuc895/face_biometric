import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

def train():
    conf = get_config(False)
    mtcnn = MTCNN()
    #print('mtcnn loaded')
    learner = face_learner(conf, True)
    learner.load_state(conf, 'cpu_final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    targets, names = prepare_facebank(conf, learner.model, mtcnn)
    return {'_result': 'success', '_message': 'Model Is Updated'}

print(train())