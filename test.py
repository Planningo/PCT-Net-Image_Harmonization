import cv2
from iharm.inference.evaluation import evaluate
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
import torch
import os
import numpy as np



image=cv2.imread('/home/ubuntu/DucoNet-Image-Harmonization/composite/generated_image_74e9aa4a-c848-493d-9f2d-a14c923e008d.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread('/home/ubuntu/DucoNet-Image-Harmonization/mask/generated_image_74e9aa4a-c848-493d-9f2d-a14c923e008d.png')
mask = mask[:, :, 0].astype(np.float32) / 255.
device = torch.device(0)
checkpoint_path = find_checkpoint('', './pretrained_models/PCTNet_ViT.pth')
net = load_model('ViT_pct', checkpoint_path, verbose=False)
use_attn = False
normalization ={'mean': [0,0,0], 'std':[1,1,1]}
predictor = Predictor(net, device, with_flip=False, hsv=False, use_attn=use_attn, 
                            mean=normalization['mean'], std=normalization['std'])
 
pred_fullres = evaluate(image,mask,predictor)
cv2.imwrite(os.path.join('/home/ubuntu/PCT-Net-Image-Harmonization/output', 'harmonized.png'), pred_fullres[:,:,::-1])
