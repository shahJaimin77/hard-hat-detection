import os
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# model = torch.hub.load('ultralytics/yolov5', 'custom',
#                     path='best_hh.pt', force_reload=True)

from pathlib import Path

yolo_path = r"{}\yolov5".format(os.getcwd())
print(yolo_path)

model = torch.hub.load(yolo_path, 'custom', path='best_hh.pt', source='local')
# Image

def obj_detection(my_img):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    column1, column2 = st.columns(2)

    column1.subheader("Input image")
    st.text("")
    plt.figure(figsize = (16,16))
    plt.imshow(my_img)
    column1.pyplot(use_container_width = True)

    # with open("C://Users//lenovo//Documents//Downloads//coco.names", "r") as f:
    #     labels = [line.strip() for line in f.readlines()]
    # names_of_layer = net.getLayerNames()
    # output_layers = [names_of_layer[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(1, 3))   


    # Image loading
    newImage = np.array(my_img.convert('RGB'))
    img = cv2.cvtColor(newImage,1)

    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)

    # indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      
    # print(indexes)
    model.conf = score_threshold
    model.iou = nms_threshold

    results = model(img)
    cv2.imshow('Fire Detection', np.squeeze(results.render()))

    st.text("")
    column2.subheader("Output image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(np.squeeze(results.render()))
    column2.pyplot(use_container_width = True)

    # if len(indexes)>1:
    #     st.success("Found {} Objects - {}".format(len(indexes),[item for item in set(items)]))
    # else:
    #     st.success("Found {} Object - {}".format(len(indexes),[item for item in set(items)]))


def main():
    
    st.title("Welcome to Nazar AI")
    st.write("You can view real-time object detection done. Select one of the following options to proceed:")

    choice = st.radio("Perform hard-hat detection", ("See an illustration", "Choose an image of your choice"))
    #st.write()

    if choice == "Choose an image of your choice":
        #st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])

        if image_file is not None:
            my_img = Image.open(image_file)  
            obj_detection(my_img)

    elif choice == "See an illustration":
        my_img = Image.open("test_images/1.jpeg")
        obj_detection(my_img)

if __name__ == '__main__':
    main()

