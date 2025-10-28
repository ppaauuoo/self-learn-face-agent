import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Method-1, use FaceAnalysis
app = FaceAnalysis(allowed_modules=["detection"])  # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

# Method-2, load model directly
detector = insightface.model_zoo.get_model("buffalo_l")
detector.prepare(ctx_id=0, input_size=(640, 640))
