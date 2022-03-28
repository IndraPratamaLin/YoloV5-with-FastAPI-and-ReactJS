from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

import cv2
import numpy as np

import torch
import base64
import random

# try-yolo
# C:\Users\GDA-Users\Desktop\yolo-backend\backend

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = torch.hub.load("ultralytics/yolov5", 'yolov5s', pretrained=True)


colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting


def results_to_json(results, model):
	''' Converts yolo model output to json (list of list of dicts)'''
	return [
				[
					{
					"class": int(pred[5]),
					"class_name": model.model.names[int(pred[5])],
					"bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
					"confidence": float(pred[4]),
					}
				for pred in result
				]
			for result in results.xyxy
			]



def base64EncodeImage(img):
	''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
	_, im_arr = cv2.imencode('.jpg', img)
	im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

	return im_b64



# To show bounding boxes on image
def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
	# Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# Setting up the home route
@app.get("/")
def read_root():
    return {"data": "Welcome to YoloV5 Object Recognition API"}


# detect via API
@app.post("/detect/")
async def detect_via_api(file: UploadFile = File(...), img_size: Optional[int] = Form(640), download_image: Optional[bool] = Form(False)):
	

	model = torch.hub.load("ultralytics/yolov5", 'yolov5s', pretrained=True)

	image = cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)

	#create a copy that corrects for cv2.imdecode generating BGR images instead of RGB, 
	#using cvtColor instead of [...,::-1] to keep array contiguous in RAM
	img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	results = model(img_rgb, size = img_size) 
	json_results = results_to_json(results, model)

	if download_image:

		for bbox in json_results[0]:
			label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
			plot_one_box(bbox['bbox'], image, label=label, color=colors[int(bbox['class'])], line_thickness=3)

		payload = {'image_base64':base64EncodeImage(image)}
		json_results.append(payload)

	# debugging
	# json_results.append({'try': len(json_results[0])})

	return json_results



