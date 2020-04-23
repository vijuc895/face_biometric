import cv2
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from multiprocessing import Process, Pipe,Value,Array
import torch
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank,load_images_from_folder
#parser = argparse.ArgumentParser(description='take a picture')
#parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
#args = parser.parse_args()
from config import get_config
from pathlib import Path
def register(user_id):
    data_path = Path('data')
    save_path = data_path/'facebank'/user_id
    fetch_path=data_path/'dataset'/user_id
    images=load_images_from_folder(fetch_path)
    print(images)
    if not save_path.exists():
        save_path.mkdir()

    mtcnn = MTCNN()
    count =0
    face_id=user_id
    count =0
    for img in images:
        frame =img
        p =  Image.fromarray(frame[...,::-1])
        try:            
            warped_face = np.array(mtcnn.align(p))[...,::-1]
            cv2.imwrite("data/facebank/" + str(face_id)+'/'+str(face_id) + '_' + str(count) + ".jpg", warped_face)
            count+=1
            #cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face)
        except:
            result={"_result":"Error", "_message":"Unable to detect the face"}
    if count==len(images):
            result={"_result":"success", "_message":"User Registered Successfully"}

    conf = get_config(False)
    learner = face_learner(conf, True)
    learner.load_state(conf, 'cpu_final.pth', True, True)
    learner.model.eval()
    #print('learner loaded')
    targets, names = prepare_facebank(conf, learner.model, mtcnn,user_id)
    #print('facebank updated')
    return result
print(register("varun"))
