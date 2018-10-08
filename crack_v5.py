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


img_height = 300
img_width = 300

K.clear_session() 

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

# 학습된 weight의 경로를 지정
weights_path = '/usr/local/lib/python3.5/dist-packages/tensorflow/keras/ssd_keras/ssd300_pascal_07+12_epoch-08_loss-1.9471_val_loss-1.9156.h5'


model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 영상의 경로를 지정하고 프레임 캡쳐
# 나중에는 비디오 캡쳐를 함과 동시에 input_images리스트에 곧바로 넣어버려서, 불필요한 이미지 입출력 과정을 줄이자
# 1초에 4프레임 캡쳐로 바꿈 (6프레임마다 저장)

from wand.drawing import Drawing
from wand.image import Image
from wand.color import Color
import os
import sys

# We will get video name from node.js server this is demo version

filename = sys.argv[1]
video_path = '/home/starever222/SPARK/SPARK/public/videos/'+filename+'.mp4'
vidcap = cv2.VideoCapture(video_path)
success, imagefile = vidcap.read()
count = 0

# make directory


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

#if not os.path.exists(newcroppedpath):
#    os.makedirs(newcroppedpath)

# newcroppedpath = "home/starever222/SPARK/SPARK/public/cropped_frames/"+filename
#
# if not os.path.exists(newcroppedpath):
#     os.makedirs(newcroppedpath)
#
# newSauvolapath = "home/starever222/SPARK/SPARK/public/Sauvola/"+filename
#
# if not os.path.exists(newSauvolapath):
#     os.makedirs(newSauvolapath)
#
# newSkeletonpath = "home/starever222/SPARK/SPARK/public/Skeleton/"+filename
#
# if not os.path.exists(newSkeletonpath):
#     os.makedirs(newSkeletonpath)
#
# newedgespath = "home/starever222/SPARK/SPARK/public/edges/"+filename
#
# if not os.path.exists(newedgespath):
#     os.makedirs(newedgespath)



while success:

    if (count % 6 == 0):
        # 프레임 캡쳐를 저장할 경로
        # img_path = "~/SPARK/SPARK/public/images/"+filename+"/%d.jpg" % count
        cv2.imwrite("/home/starever222/SPARK/SPARK/public/images/"+filename+"/%d.jpg" % count, imagefile)
        # print("write", img_path, imagefile)
    success, imagefile = vidcap.read()
    count += 1

orig_images = []
input_images = []

# range는 추후 영상 플레이 타임 정보를 사용할 수 있도록 변경
frames_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(0,frames_count):
    if (i % 6 == 0):
        # 프레임 캡쳐를 불러오는 경로
        # ---------------------------
        # img_path = 'images/video/%d.jpg' % i
        # img_path = 'C:\\Users\\rlaal\\Desktop\\frame\\frame%d.jpg'%i
        # ----------------------------done
        # print(img_path)
        img_path = '/home/starever222/SPARK/SPARK/public/images/'+filename+'/%d.jpg' % i
        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        img = np.array(img)
        input_images.append(img)

input_images = np.array(input_images)
orig_images = np.array(orig_images)

num_of_frames = 16
counting = 0
saving_bounding_boxes = []
isBreak = 0;

print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')

# range는 추후 변수를 사용할 수 있도록 변경
for i in range(0, frames_count):
    y_pred = model.predict(input_images[i * num_of_frames:i * num_of_frames + num_of_frames])
    confidence_threshold = 0.4

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    for j in range(0, num_of_frames):
        print('frame :', counting)
        #   print(y_pred_thresh[j])
        if(j>len(y_pred_thresh)): break;
        for box in y_pred_thresh[j]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * orig_images[0].shape[1] / img_width
            ymin = box[3] * orig_images[0].shape[0] / img_height
            xmax = box[4] * orig_images[0].shape[1] / img_width
            ymax = box[5] * orig_images[0].shape[0] / img_height
            print('xmin : ', xmin, '  ymin : ', ymin, '  xmax : ', xmax, '  ymax : ', ymax)
            # 균열이 탐지된 프레임과 b-box 정보가 saving_bounding_boxes <- 여기에 저장됨
            saving_bounding_boxes.append([(counting), xmin, ymin, xmax, ymax])
            # --------------------------------------
            #framepath = "images/video_crack/frame%d.jpg" % (count), imagefile
            cv2.imwrite("/home/starever222/SPARK/SPARK/public/images/"+filename+"_crack/%d.jpg" % (counting), imagefile)
            #add drawing rectangle
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
                #with Image(filename="/home/starever222/SPARK/SPARK/public/images/"+filename+"/%d.jpg"% count) as image:
                with Image(filename=img_path) as image:
                    draw(image)
                    #boximg_path = "/home/starever222/SPARK/SPARK/public/images/"+filename+"/%d.jpg"% count
                    boximg_path = "/home/starever222/SPARK/SPARK/public/images/"+filename+"/%d.jpg" % counting
                    image.save(filename=boximg_path)
                    #cv2.imwrite("C:\\Users\\rlaal\\Desktop\\detected\\frame%d.jpg"% (count), imagefile)
                    # ----------------------------------------done

        counting += 6
        if(counting>frames_count): 
            isBreak = 1;
            break;
    if(isBreak == 1): break;

print(saving_bounding_boxes)

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
    # -----------------------------
    #newcroppedpath = "/home/starever222/SPARK/SPARK/public/cropped_frames/" + filename

    #if not os.path.exists(newcroppedpath):
    #    os.chmod(0777)
    #    os.makedirs(newcroppedpath)
    # img_path = newcroppedpath+'/%d.jpg' % frame_count
    # img_path = '../../Desktop/test/%d.jpg'%frame_count
    # -----------------------done
    cropped_frames.append(cropped_frame)
    # io.imsave(img_path, cropped_frame)

# 1. Image binarization(Sauvola's method) using Pw and Pl, respectively
# 오래 걸리는 문제가 있음

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

    # 논문에선 각각 70,180이었으나 여기선 홀수 input만 가능
    window_size_Pw = 71
    thresh_sauvola_Pw = threshold_sauvola(img_gray, window_size=window_size_Pw, k=0.42)

    # Below are the converted images through Sauvola's method.
    # _bw will contain 0 or 1, not true or false. bw means black or white.
    binary_sauvola_Pw = img_gray > thresh_sauvola_Pw
    binary_sauvola_Pw_bw = img_gray > thresh_sauvola_Pw

    binary_sauvola_Pw_bw.dtype = 'uint8'

    binary_sauvola_Pw_bw *= 255

    sauvola_frames_Pw_bw.append(binary_sauvola_Pw_bw)
    sauvola_frames_Pw.append(binary_sauvola_Pw)
    # ----------------------------------------
    # newSauvolapath = "/home/starever222/SPARK/SPARK/public/Sauvola/" + filename

    # if not os.path.exists(newSauvolapath):
    #    os.makedirs(newSauvolapath)
    # img_path_Pw = newSauvolapath+'/Sauvola_Pw_%d.jpg' % i
    #   img_path_Pw = '../../Desktop/Sauvola/Sauvola_Pw_%d.jpg'%i
    # -------------------------------------------done
    # io.imsave(img_path_Pw, binary_sauvola_Pw_bw)

# 2. Extract the skeletons of each images

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

    skeleton_frames_Pw.append(skeleton_Pw)
    # ---------------------------
    #newSkeletonpath = "/home/starever222/SPARK/SPARK/public/Skeleton/" + filename

    #if not os.path.exists(newSkeletonpath):
    #    os.chmod(0777)
    #    os.makedirs(newSkeletonpath, 0777)
    # img_path_Pw = newSkeletonpath+"/skeleton_Pw_%d.jpg" % i
    # img_path_Pw = "../../Desktop/Skeleton/skeleton_Pw_%d.jpg"%i
    # ---------------------------done
    # io.imsave(img_path_Pw, skeleton_Pw)

# 3. Detect the edges of each images
### edge detection 할 때, 좋은 parameter를 찾아야 한다. 지금은 edge가 너무 두꺼움 (overestimation됨) ###
import numpy as np
from scipy import ndimage as ndi
from skimage import feature

edges_frames_Pw = []

for i in range(0,len(cropped_frames)):
    # Compute the Canny filter for two values of sigma
    # canny(image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)
    # sigma가 1이었으나, 0.1로 조정하여 실제 균열 edge와 거의 같게 만듦.
    # 정확도에서 문제가 생긴다면 1. skeleton의 방향 설정 방법을 바꾸던가, 2. 여기서 시그마 값을 살짝 늘리거나 줄여가면서 정확도를 테스트 해볼 것
    edges_Pw = feature.canny(sauvola_frames_Pw[i], 0.09)

    edges_Pw.dtype = 'uint8'

    edges_Pw *= 255

    edges_frames_Pw.append(edges_Pw)
    # ----------------------------
    # newedgespath = "/home/starever222/SPARK/SPARK/public/edges/" + filename

    # if not os.path.exists(newedgespath):
    #     os.chmod(newedgespath, 0777)
    #    os.makedirs(newedgespath)
    # img_path_Pw = newedgespath+"/edges_Pw_%d.jpg"%i
    # img_path_Pw = "../../Desktop/edges/edges_Pw_%d.jpg"%i
    # ----------------------------done
    # io.imsave(img_path_Pw, edges_Pw)

# Crack만이 detection되어서 넘어왔다는 가정이 있어야 함. 아니면 외부 배경 이미지도 균열 계산에 포함 됨

# 7. 균열의 폭을 계산합니다. 
# 1) 균열의 Skeleton으로부터 균열의 진행 방향을 파악합니다.
# 2) 균열의 진행 방향에 수직인 직선을 긋습니다.
# 3) 이 수직선과 균열의 Edge가 만나는데, 이 거리가 곧 균열의 폭입니다.

# 7. Calculate the width of the crack.
# 1) Analyze the direction of the crack from the skeleton.
# 2) Draw a perpendicular line of the direction
# 3) The perpendicular line meets the edge. This distance is the width of the crack.

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
    if(len(crack_width_list)==0): continue;
    if(len(crack_width_list)<10):
        real_width = round(crack_width_list[len(crack_width_list)-1]*0.92, 2)
    else: real_width = round(crack_width_list[9]*0.92, 2)
    
    save_result.append(real_width)

    if(real_width >= 0.3):
        save_risk.append('상')
        print('위험군 : 상\n')
    elif(real_width<0.3 or real_width>=0.2): 
        save_risk.append('중')
        print('위험군 : 중\n')
    else: 
        save_risk.append('하')
        print('위험군 : 하\n')
        
f = open("/home/starever222/SPARK/SPARK/public/logs/"+filename+"/width.txt", 'w')
for z in range(0, len(skeleton_frames_Pw)):
    f.write(str(save_result[z])+'mm\n')
f.close()
fr = open("/home/starever222/SPARK/SPARK/public/logs/"+filename+"/risk.txt", 'w')
for z in range(0, len(skeleton_frames_Pw)):
    fr.write(save_risk[z]+'\n')
fr.close()
