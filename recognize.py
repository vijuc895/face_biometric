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
accuracy=[]
user=[]
def recognize(path):
    conf = get_config(False)
    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = 1.35
    learner.load_state(conf, 'cpu_final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    targets, names = load_facebank(conf)
    print(names)
    print('facebank loaded')
    count =0
    while True:
        frame = cv2.imread(path)
        try:
            image = Image.fromarray(frame)
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice    
            results, score = learner.infer(conf, faces, targets)
            
            for idx,bbox in enumerate(bboxes):
                    frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(100-score[idx]), frame)
                    result={"_result":"success", "User Verified with":{"confidence": '{:.2f}%'.format(100-score[idx]), "userid": names[results[idx] + 1] , "error": "Success"}}
                    accuracy.append('{:.2f}'.format(100-score[idx]))
                    user.append(names[results[idx] + 1])
                    print( names[results[idx] + 1],'{:.2f}'.format(100-score[idx]))
            count=1     
        except:
            print('detect error')    
        if count>0:
            break
    return result

print(recognize("test/"+str(13)+".jpg"))
print("********************************")
print("Recognized User's List is ")
print(user)
print()
print("Individual accuracy")
print(accuracy)
print()
sum=0
for i in accuracy:
    sum+=float(i)
print("The Average Accuracy is ",sum/len(accuracy))