# 1. 사전에 학습시킨 균열 탐지 딥러닝 weight와 Single Shot Multibox Detector model을 불러옵니다. 

# 1. Upload the pre-trained deep learning weight and Single Shot Multibox Detector model for crack detection.

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import os

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


img_height = 300  # Height of the model input images
img_width = 300 # Width of the model input images

K.clear_session() 

# 변수 값은 Single Shot Multibox Detector의 원래 수치를 변경하지 않고 사용하였습니다.
# The original value of parameters of 'Single Shot Multibox Detector' was used without any changes.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=2,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load some weights into the model.

# 학습된 weight를 불러오는 경로를 입력합니다.
# Input your own path for pre-trained weight.
weights_path = '/usr/local/lib/python3.5/dist-packages/tensorflow/keras/ssd_keras/ssd300_pascal_07+12_epoch-08_loss-1.9471_val_loss-1.9156.h5'


model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


# 2. 드론이 촬영한 콘크리트 외벽 영상에서 프레임을 추출합니다(4fps).
#    이 프레임 이미지들을 균열 탐지 딥러닝 엔진에 입력하여 inference를 합니다.
#    Inference의 결과 값으로 균열의 위치를 bounding box의 형태로 report합니다.

# 2. Extract frames out of the video which recorded the concrete surface shoot by drone(4fps).
#    Input the frame images into the deep learning engine for inference.
#    The positional information will be reported as a bounding box, as a result of the inference.


# 드론으로 촬영한 영상의 경로를 입력합니다.
# Input the path of the video shoot by drone.

from wand.drawing import Drawing
from wand.image import Image
from wand.color import Color
import os
import sys

# We will get video name from node.js server this is demo version


filename = sys.argv[1]  # get video's name from node.js file
video_path = '/home/starever222/SPARK/SPARK/public/videos/'+filename+'.mp4'
vidcap = cv2.VideoCapture(video_path)
success, imagefile = vidcap.read()
count = 0


# make images' and logs' directory

newimagepath = "/home/starever222/SPARK/SPARK/public/images/"+filename

if not os.path.exists(newimagepath):
    os.makedirs(newimagepath)
newframepath = "/home/starever222/SPARK/SPARK/public/images/"+filename+"_crack"

if not os.path.exists(newframepath):
    os.makedirs(newframepath)

newlogpath = "/home/starever222/SPARK/SPARK/public/logs/"+filename

if not os.path.exists(newlogpath):
    os.makedirs(newlogpath)

#newcroppedpath = "/home/starever222/SPARK/SPARK/public/cropped_frames/"+filename
#
#if not os.path.exists(newcroppedpath):
#    os.makedirs(newcroppedpath)
#
#newSauvolapath = "/home/starever222/SPARK/SPARK/public/Sauvola/"+filename
#
#if not os.path.exists(newSauvolapath):
#    os.makedirs(newSauvolapath)
#
#newSkeletonpath = "/home/starever222/SPARK/SPARK/public/Skeleton/"+filename
#
#if not os.path.exists(newSkeletonpath):
#    os.makedirs(newSkeletonpath)
#
# newedgespath = "/home/starever222/SPARK/SPARK/public/edges/"+filename
#
#if not os.path.exists(newedgespath):
#    os.makedirs(newedgespath)



while success:

    if (count % 6 == 0):
        # 추출된 프레임 이미지들을 저장할 경로를 입력합니다.
        # Input the path to save the extracted frame images.
        cv2.imwrite("/home/starever222/SPARK/SPARK/public/images/"+filename+"/%d.jpg" % count, imagefile)
    success, imagefile = vidcap.read()
    count += 1

orig_images = []
input_images = []

frames_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(0,frames_count):
    if (i % 6 == 0):
        # 저장한 프레임 이미지들을 다시 불러올 수 있도록 동일한 경로를 입력합니다.
        # Input the same path used before to load the saved frame images.
        img_path = '/home/starever222/SPARK/SPARK/public/images/'+filename+'/%d.jpg' % i
        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        img = np.array(img)
        input_images.append(img)

input_images = np.array(input_images)
orig_images = np.array(orig_images)

# 한 번에 처리되는 프레임 이미지의 갯수를 입력합니다. 보유 GPU 성능에 따라 늘리거나 줄여야 할 수 있습니다. 
# Input the number of frame images to be batch-processed.
# You may increase or decrease the number depending on the performance of your own gpu.
num_of_frames = 16
counting = 0
saving_bounding_boxes = []
isBreak = 0;

print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')


for i in range(0, frames_count):
    y_pred = model.predict(input_images[i * num_of_frames:i * num_of_frames + num_of_frames])
    confidence_threshold = 0.4

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    for j in range(0, num_of_frames):
        print('frame :', counting)
        if(j>len(y_pred_thresh)): break;
        for box in y_pred_thresh[j]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * orig_images[0].shape[1] / img_width
            ymin = box[3] * orig_images[0].shape[0] / img_height
            xmax = box[4] * orig_images[0].shape[1] / img_width
            ymax = box[5] * orig_images[0].shape[0] / img_height
            print('xmin : ', xmin, '  ymin : ', ymin, '  xmax : ', xmax, '  ymax : ', ymax)
            # 균열이 탐지된 프레임의 bounding box 위치정보를 saving_bounding_boxes 리스트에 저장합니다.
            # Append the positional information of the bounding box of the detected frame at'saving_bounding_boxes' list.
            
            saving_bounding_boxes.append([(counting), xmin, ymin, xmax, ymax])
            cv2.imwrite("/home/starever222/SPARK/SPARK/public/images/"+filename+"_crack/%d.jpg" % (counting), imagefile)
            
            #add drawing red rectangle on cracked images (by Wand)
            with Drawing() as draw:
                draw.stroke_width = 4.0
                draw.stroke_color = Color('red')
                draw.fill_color = Color('transparent')
                xMin = int(xmin)
                xMax = int(xmax)
                yMin = int(ymin)
                yMax = int(ymax)
                draw.rectangle(left=xMin, top=yMin, right=xMax, bottom=yMax)
                img_path = "/home/starever222/SPARK/SPARK/public/images/"+filename+"/%d.jpg" % counting
                with Image(filename=img_path) as image:
                    draw(image)
                    boximg_path = "/home/starever222/SPARK/SPARK/public/images/"+filename+"/%d.jpg" % counting
                    image.save(filename=boximg_path)

        counting += 6
        if(counting>frames_count): 
            isBreak = 1;
            break;
    if(isBreak == 1): break;

print(saving_bounding_boxes)

# 3. 균열탐지 딥러닝 엔진이 리포트 한 균열 위치에 맞게 프레임 이미지를 잘라냅니다.
# 3. Crop the frame image using the positional information of the crack reported by crack detection deep learning engine.
from skimage import io

cropped_frames = []

for i in range(0, len(saving_bounding_boxes)):
    frame_count = saving_bounding_boxes[i][0]//6
    frame = orig_images[frame_count]
    if(saving_bounding_boxes[i][1] < 0):
        saving_bounding_boxes[i][1] = 0
    xmin = int(saving_bounding_boxes[i][1])
    if(saving_bounding_boxes[i][2] < 0):
        saving_bounding_boxes[i][2] = 0
    ymin = int(saving_bounding_boxes[i][2])
    xmax = int(saving_bounding_boxes[i][3])
    ymax = int(saving_bounding_boxes[i][4])
    print(xmin,ymin,xmax,ymax)
    cropped_frame = orig_images[frame_count][ymin:ymax, xmin:xmax, :]
    cropped_frame = cropped_frame.astype('uint8')
    
    # 잘라낸 프레임 이미지를 저장하는 리스트입니다.
    # The list which saves the cropped frame images.
    
    cropped_frames.append(cropped_frame)

# 4. 균열 탐지 딥러닝 엔진이 잘라낸 프레임 이미지에 전처리를 합니다.
#    전처리는 총 3단계로 구성됩니다.
#   1) Image Binarization : 균열인 부분과 균열이 아닌 부분을 분리합니다.
#   2) Skeletonize : 균열의 뼈대를 추출합니다.
#   3) Edge detection : 균열의 외곽선을 추출합니다.

#   이 단계에서는 Image Binarization을 진행합니다.

# 4. Preprocess the frame images cropped by crack detection deep learning engine.
#    The preprocess consists of 3 stages.
#   1) Image Binarization : seperate crack section and the noncrack section.
#   2) Skeletonize : extract the central skeleton of the crack.
#   3) Edge detection : extract the edge of the crack.

#   At this stage, Image Binarization will be done.

import time
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage import data
from skimage.color import rgb2gray
from skimage.data import page
from skimage.filters import (threshold_sauvola)
from PIL import Image

sauvola_frames_Pw_bw = []
sauvola_frames_Pw = []

# Upload the image
for i in range(0, len(cropped_frames)):
    img = cropped_frames[i]
    img_gray = rgb2gray(img)

    # window size와 k값은 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing' 논문이 제시한 값을
    # 그대로 사용하였습니다.
    
    # window size and k value were used without any changes from the
    # 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing' thesis.
    window_size_Pw = 71
    thresh_sauvola_Pw = threshold_sauvola(img_gray, window_size=window_size_Pw, k=0.42)

    # Below are the converted images through Sauvola's method.
    # _bw will contain 0 or 1, not true or false. bw means black or white.
    binary_sauvola_Pw = img_gray > thresh_sauvola_Pw
    binary_sauvola_Pw_bw = img_gray > thresh_sauvola_Pw

    binary_sauvola_Pw_bw.dtype = 'uint8'

    binary_sauvola_Pw_bw *= 255
    
    # Image Binarization이 완료된 이미지를 저장하는 리스트입니다.
    # The list which saves the images after image binarization.
    
    sauvola_frames_Pw_bw.append(binary_sauvola_Pw_bw)
    sauvola_frames_Pw.append(binary_sauvola_Pw)

# 5. 균열의 뼈대를 추출합니다.
# 5. Extract the skeleton of the crack.

from skimage.morphology import skeletonize
from skimage.util import invert

skeleton_frames_Pw = []

for i in range(0, len(cropped_frames)):
    # Invert the binarized images
    img_Pw = invert(sauvola_frames_Pw[i])

    # Below are skeletonized images
    skeleton_Pw = skeletonize(img_Pw)

    # Convert true/false to 1/0 to save it as image
    skeleton_Pw.dtype = 'uint8'

    skeleton_Pw *= 255
    
    # Skeletonize가 끝난 이미지를 저장하는 리스트입니다.
    # The list which saves the images after the skeletonization.
    skeleton_frames_Pw.append(skeleton_Pw)

# 6. 균열의 외곽선을 추출합니다.
# 6. Detect the edges of the crack.

import numpy as np
from scipy import ndimage as ndi
from skimage import feature

edges_frames_Pw = []

for i in range(0,len(cropped_frames)):
    # Compute the Canny filter for two values of sigma
    # canny(image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)
    edges_Pw = feature.canny(sauvola_frames_Pw[i], 0.09)

    edges_Pw.dtype = 'uint8'

    edges_Pw *= 255
    
    # Edge detection이 끝난 이미지를 저장하는 리스트입니다.
    # The list which saves the images after edge detection.
    edges_frames_Pw.append(edges_Pw)

# 7. 균열의 폭을 계산합니다. 
# 1) 균열의 Skeleton으로부터 균열의 진행 방향을 파악합니다.
# 2) 균열의 진행 방향에 수직인 직선을 긋습니다.
# 3) 이 수직선과 균열의 Edge가 만나는데, 이 거리가 곧 균열의 폭입니다.

# 7. Calculate the width of the crack.
# 1) Analyze the direction of the crack from the skeleton.
# 2) Draw a perpendicular line of the direction
# 3) The perpendicular line meets the edge. This distance is the width of the crack..

import queue
import math

dx_dir_right = [-5,-5,-5,-4,-3,-2,-1,0,1,2,3,4,5,5]
dy_dir_right = [0,1,2,3,4,5,5,5,5,5,4,3,2,1]

dx_dir_left = [5,5,5,4,3,2,1,0,-1,-2,-3,-4,-5,-5]
dy_dir_left = [0,-1,-2,-3,-4,-5,-5,-5,-5,-5,-4,-3,-2,-1]

dx_bfs = [-1,-1,0,1,1,1,0,-1]
dy_bfs = [0,1,1,1,0,-1,-1,-1]

save_result = []
save_risk = []

# BFS를 통해 Skeleton을 찾습니다.
# Searching the skeleton through BFS.
for k in range(0,len(skeleton_frames_Pw)):
    print('--------------''동영상 내 재생 시간 : ',(saving_bounding_boxes[k][0]//6)*0.25,'초','-----------------')
    start = [0,0]
    next = []
    q = queue.Queue()
    q.put(start)

    len_x = skeleton_frames_Pw[k].shape[0]
    len_y = skeleton_frames_Pw[k].shape[1]

    visit = np.zeros((len_x,len_y))
    crack_width_list = []

    while(q.empty() == 0):
        next = q.get()
        x = next[0]
        y = next[1]
        right_x = right_y = left_x = left_y = -1

        if(skeleton_frames_Pw[k][x][y] == 255):
            # Skeleton을 바탕으로 균열의 진행 방향을 구합니다.
            # Estimating the direction of the crack from skeleton
            for i in range(0, len(dx_dir_right)):
                right_x = x + dx_dir_right[i]
                right_y = y + dy_dir_right[i]
                if(right_x<0 or right_y<0 or right_x>=len_x or right_y>=len_y): 
                    right_x = right_y = -1
                    continue;
                if(skeleton_frames_Pw[k][right_x][right_y] == 255): break;
                if(i==13): right_x = right_y = -1

            if(right_x == -1): 
                right_x = x
                right_y = y

            for i in range(0, len(dx_dir_left)):
                left_x = x + dx_dir_left[i]
                left_y = y + dy_dir_left[i]
                if(left_x <0 or left_y<0 or left_x >=len_x or left_y>=len_y): 
                    left_x = left_y = -1
                    continue;
                if(skeleton_frames_Pw[k][left_x][left_y] == 255): break;
                if(i==13): left_x = left_y = -1

            if(left_x == -1): 
                left_x = x
                left_y = y

            base = right_y - left_y
            height = right_x - left_x
            hypotenuse = math.sqrt(base*base + height*height)

            if(base==0 and height != 0): theta = 90.0
            elif(base==0 and height == 0): continue
            else: theta = math.degrees(math.acos((base * base + hypotenuse * hypotenuse - height * height)/(2.0 * base * hypotenuse)))

            theta += 90
            dist = 0
            
            # 균열 진행 방향의 수직선과 Edge가 만나면, 그 거리를 구합니다.
            # Calculate the distance if the perpendicular line meets the edge of the crack.
            for i in range(0,2):
                
                pix_x = x
                pix_y = y
                if(theta>360): theta -= 360
                elif(theta<0): theta += 360    
                
                if(theta == 0.0 or theta == 360.0):
                    while(1):
                        pix_y+=1
                        if(pix_y>=len_y):
                            pix_x = x
                            pix_y = y
                            break;
                        if(edges_frames_Pw[k][pix_x][pix_y]==255): break;

                elif(theta == 90.0):
                    while(1):
                        pix_x-=1
                        if(pix_x<0):
                            pix_x = x
                            pix_y = y
                            break;
                        if(edges_frames_Pw[k][pix_x][pix_y]==255): break;

                elif(theta == 180.0):
                    while(1):
                        pix_y-=1
                        if(pix_y<0):
                            pix_x = x
                            pix_y = y
                            break;
                        if(edges_frames_Pw[k][pix_x][pix_y]==255): break;

                elif(theta == 270.0):
                    while(1):
                        pix_x+=1
                        if(pix_x>=len_x):
                            pix_x = x
                            pix_y = y
                            break;
                        if(edges_frames_Pw[k][pix_x][pix_y]==255): break;
                else:
                    a = 1
                    radian = math.radians(theta)
                    while(1):        
                        pix_x = x - round(a*math.sin(radian))
                        pix_y = y + round(a*math.cos(radian))
                        if(pix_x<0 or pix_y<0 or pix_x>=len_x or pix_y>=len_y): 
                            pix_x=x
                            pix_y=y
                            break;
                        if(edges_frames_Pw[k][pix_x][pix_y]==255): break;

                        if(theta>0 and theta<90):
                            if(pix_y+1<len_y and edges_frames_Pw[k][pix_x][pix_y+1]==255): 
                                pix_y+=1
                                break;
                            if(pix_x-1>=0 and edges_frames_Pw[k][pix_x-1][pix_y]==255): 
                                pix_x-=1
                                break;

                        elif(theta>90 and theta<180):
                            if(pix_y-1>=0 and edges_frames_Pw[k][pix_x][pix_y-1]==255): 
                                pix_y-=1
                                break;
                            if(pix_x-1>=0 and edges_frames_Pw[k][pix_x-1][pix_y]==255): 
                                pix_x-=1
                                break;

                        elif(theta>180 and theta<270):
                            if(pix_y-1>=0 and edges_frames_Pw[k][pix_x][pix_y-1]==255): 
                                pix_y-=1
                                break;
                            if(pix_x+1<len_x and edges_frames_Pw[k][pix_x+1][pix_y]==255): 
                                pix_x+=1
                                break;         

                        elif(theta>270 and theta<360): 
                            if(pix_y+1<len_y and edges_frames_Pw[k][pix_x][pix_y+1]==255): 
                                pix_y+=1
                                break;
                            if(pix_x+1<len_x and edges_frames_Pw[k][pix_x+1][pix_y]==255): 
                                pix_x+=1
                                break;
                        a+=1
        
                dist += math.sqrt((y-pix_y)**2 + (x-pix_x)**2)
                theta += 180        

            # 균열의 폭을 저장하는 리스트입니다.
            # The list which saves the width of the crack.
            crack_width_list.append(dist)
        
        for i in range(0,8):
            next_x = x + dx_bfs[i]
            next_y = y + dy_bfs[i]

            if(next_x<0 or next_y<0 or next_x>=len_x or next_y>=len_y): continue;
            if(visit[next_x][next_y] == 0): 
                q.put([next_x,next_y])
                visit[next_x][next_y] = 1
                
    crack_width_list.sort(reverse=True)
    if(len(crack_width_list)==0): 
        save_result.append(0)
        real_width = 0
    elif(len(crack_width_list)<10):
        real_width = round(crack_width_list[len(crack_width_list)-1]*0.92, 2)
        save_result.append(real_width)
    else: 
        real_width = round(crack_width_list[9]*0.92, 2)
        save_result.append(real_width)
    # give level of risk data in save_risk
    if(real_width >= 0.3):
        save_risk.append('상')
        print('위험군 : 상\n')
    elif(real_width<0.3 and real_width>=0.2): 
        save_risk.append('중')
        print('위험군 : 중\n')
    else: 
        save_risk.append('하')
        print('위험군 : 하\n')

# make width and level of risk data to text data
f = open("/home/starever222/SPARK/SPARK/public/logs/"+filename+"/width.txt", 'w')
for z in range(0, len(save_result)):
    f.write(str(save_result[z])+'mm\n')
f.close()
fr = open("/home/starever222/SPARK/SPARK/public/logs/"+filename+"/risk.txt", 'w')
for z in range(0, len(save_risk)):
    fr.write(save_risk[z]+'\n')
fr.close()
