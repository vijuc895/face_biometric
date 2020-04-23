# InsightFace_Pytorch

Pytorch0.4.1 codes for InsightFace

------

## 1. Intro

- This repo is a reimplementation of Arcface[(paper)](https://arxiv.org/abs/1801.07698), or Insightface[(github)](https://github.com/deepinsight/insightface)
- For models, including the pytorch implementation of the backbone modules of Arcface and MobileFacenet
- Codes for transform MXNET data records in Insightface[(github)](https://github.com/deepinsight/insightface) to Image Datafolders are provided
- Pretrained models are posted, include the [MobileFacenet](https://arxiv.org/abs/1804.07573) and IR-SE50 in the original paper

------|

## 2. How to use

- clone

  ```
  git clone https://github.com/vijuc895/face_biometric
  ```

- Move to this directory
  
  cd face_biometric

- Add pretrained model 

  1. model_ir_se50.pth to workspace/models/
  2. model_cpu_final.pth to workspace/save/

  I will use these two model.

- After this everything is well setuped, now we can move to execution.

1. run python3 update.py to train the model on the images in data/facebank dir

2. python3 register.py if you need to register any face

3. python3 recognize.py for recognition purpose

4. python3 authenticuser for authentication purpose.
