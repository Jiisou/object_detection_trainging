# Inference for ONNX model
import os # added by me; 간단한 개발 또는 편집이면 여기서 해도 되고,, 그렇지 않으면 로컬에서 

import cv2
cuda = True
w = "yolov7-tiny.onnx"
#img = cv2.imread('horses.jpg')  # image-based execute!

import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

providers = ['AzureExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
