from base64 import b64decode
import base64
import io
import asyncio
import os
# from image_check.has_human import has_human
from fastapi.exceptions import RequestValidationError
from concurrent.futures import ThreadPoolExecutor

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch, gc
from typing import Annotated,Union
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Form
from fastapi.responses import Response, JSONResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import cv2
import numpy as np
from iharm.inference.evaluation import evaluate
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
import torch

cors_options = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    "allow_credentials": True,
    "allow_origins": [
        "http://localhost:3000",
        "http://localhost",
        "https://dev-app.photio.io",
        "https://app.photio.io",
        "https://cafe24.photio.io"
    ],
}


app = FastAPI()

app.add_middleware(CORSMiddleware, **cors_options)

@app.get("/")
def read_root():
    return {"Hello": "World!"}

@app.get("/ping")
def read_root():
    return "pong"


def decode_base64_to_image(encoding: str) -> Image.Image:
    if encoding.startswith("http://") or encoding.startswith("https://"):
        try:
            response = requests.get(encoding, timeout=30, verify=False)
            return Image.open(io.BytesIO(response.content))
        except requests.exceptions.Timeout as e:
            raise HTTPException(status_code=408) from e
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=422) from e
        except Exception as e:
            raise HTTPException(status_code=422) from e
    else:
        if encoding.startswith("data:"):
            encoding = encoding.split(";")[1].split(",")[1]
        try:
            im_bytes = base64.b64decode(encoding)
            im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
            img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            raise HTTPException(status_code=422) from e



class HarmonizeRequest(BaseModel):
    image:str = Field(
        None,
        title="Image",
        description="base64 or url",
    )
    mask:str = Field(
        None,
        title="Image",
        description="base64 or url",
    )
    

device = torch.device(0)
checkpoint_path = find_checkpoint('', './pretrained_models/PCTNet_ViT.pth')
net = load_model('ViT_pct', checkpoint_path, verbose=False)
use_attn = False
normalization ={'mean': [0,0,0], 'std':[1,1,1]}
predictor = Predictor(net, device, with_flip=False, hsv=False, use_attn=use_attn, 
                            mean=normalization['mean'], std=normalization['std'])
 


@app.post("/harmonize")
async def harmonize(
    request:HarmonizeRequest
):
    try:
        image = decode_base64_to_image(request.image)
        mask = decode_base64_to_image(request.mask)
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pred_fullres = evaluate(image,mask,predictor)
        # cv2.imwrite(os.path.join('/home/ubuntu/PCT-Net-Image-Harmonization/output', 'harmonized.png'), pred_fullres[:,:,::-1])

        _, im_arr = cv2.imencode('.png', pred_fullres)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)


        return {"result": im_b64}
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"error occur: {e}") from e
