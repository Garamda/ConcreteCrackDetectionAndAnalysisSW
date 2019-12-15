# 드론을 활용한 콘크리트 구조물 균열 탐지 소프트웨어

</br>

## 소개
- 콘크리트 구조물을 안전 진단 할 때, 전문가들이 가장 먼저 보는 것이 외관 상태입니다. 그 중에서도 외벽상의 균열은 가장 확실한 상태판단 요소입니다. 때문에 안전 진단 시 가장 먼저 살피는 이상 징후 중 하나가 균열입니다. <br><br>
![image](https://user-images.githubusercontent.com/28426269/46650304-f8a80180-cbd6-11e8-9384-a294f25e2691.png)

<사진 1. 균열 분석 현장>

- 콘크리트 균열은 일정한 패턴을 가지고 있으므로, 안전 진단 시 소프트웨어를 통해 균열 탐지 작업을 자동화하는 것이 가능합니다. 하지만 아직까지는 주로 특수 장비를 통해서만 진단을 해 왔을 뿐입니다. 소프트웨어를 통한 콘크리트 구조물의 균열 탐지는 안전 진단에 들어가는 비용, 시간, 노력을 크게 줄여줍니다.
</br></br>

- 뿐만 아니라, 드론이 여기에 함께 사용된다면 더욱 효율적인 안전 진단이 가능합니다. 드론은 사람이 직접 닿기 어려운 부분에 쉽게 접근할 수 있기 때문입니다. 특히 대형 콘크리트 구조물을 진단할 때 드론은 더욱 유용하게 쓰입니다. 대형 구조물은 안전 진단 인력이 모든 부분을 직접 살피기 불가능하기 때문입니다. 이러한 이유로 드론은 최근들어 대형 콘크리트 구조물의 안전 진단을 위해 활발히 연구 및 활용되고 있습니다. <br><br>
<img src="https://user-images.githubusercontent.com/28426269/46650458-913e8180-cbd7-11e8-880e-ad47d539abcc.JPG" width="70%">

<사진 2. 드론을 활용한 구조물 안전 진단>

- 따라서 드론을 활용한 안전 진단에 맞는 SW가 필요합니다. 하지만 드론으로 촬영한 콘크리트 외벽 영상에서 균열을 탐지하고 분석하는 오픈소스 SW는 아직 없는 실정입니다. 이제 Chamomile에서 드론 활용 안전 진단을 위한 콘크리트 균열 탐지 SW를 사용하실 수 있습니다. 균열을 탐지하기 위한 딥러닝 알고리즘으로는 Single Shot Multibox Detector(https://github.com/pierluigiferrari/ssd_keras) 를 사용했습니다. 균열의 폭을 측정하기 위해서 Image binarization, Skeletonize, Edge detection의 전처리 방법론을 사용하였습니다.

</br>

## 기존 안전 진단의 문제점

### 1. 자동화 되지 않은 진단
데이터에 대한 모든 판단은 안전 진단 인력이 직접 해야합니다. 균열은 일정한 패턴을 가지고 있기 때문에 SW를 통한 자동화가 가능함에도, 특수 장비는 데이터를 수집하는 것만 도울 뿐입니다.
</br>

### 2. 높은 비용 
<img src="https://user-images.githubusercontent.com/28426269/46481857-3bf91d80-c82f-11e8-8a9f-718a18bb5e86.jpg" width="70%">

<사진 3. 굴절차를 사용한 구조물 안전진단> <br> <br>
교량 안전 진단을 예시로 설명하겠습니다.
<br> <br>
* **굴절차 대여 비용** : 교량 하부를 진단할 때에는 위 사진과 같이 굴절차를 사용합니다. 문제는 굴절차 1회 대여에 약 100만원 가량의 높은 비용이 발생한다는 점입니다. 드론과 Chamomile을 사용하면 초기 구매 비용만 소요됩니다.<br></br>
* **투입 인력** : 교량 안전 진단 시 평균적으로 11명의 인력이 필요합니다. 굴절차를 운전하는 인원 1명, 굴절차의 팔에 탑승하여 교량 하부를 점검하는 인원 2명, 교량 상부에서 신호 통제를 하는 인원 8명이 필요합니다. 굴절차 대여 비용과 더불어 인력 고용의 비용까지 들어가는 것입니다. 이와 같은 이유로, 10개 교량을 기준으로 평균 4천만원의 진단 비용이 발생합니다. 드론과 Chamomile을 사용한다면 드론을 운전하는 인력 1명만이 필요합니다.

</br>
</br>

## Chamomile의 목표

### 1. 균열 탐지
콘크리트 외벽상의 균열을 탐지합니다.</br></br>
### 2. 균열 폭 측정 & 심각도에 따른 분류
탐지된 균열의 폭을 측정하고, 폭이 큰 심각한 균열을 우선적으로 리포트 합니다. 균열의 폭을 측정하는 이유는 실제 안전 진단 현장에서 위험도를 판단하기 위해 사용하는 중요한 정보이기 때문입니다.  <br> 폭이 0.3mm보다 크면 유지 보수 작업에 들어가고, 0.3mm~0.2mm에 해당하면 추이를 지켜봅니다. 0.2mm 미만인 균열은 위험도가 낮다고 판단합니다. Chamomile에서는 이 기준을 그대로 사용하여, 아래와 같이 위험군을 진단합니다.

| 폭 | 위험군 |
| :---: | :---: |
| > 0.3mm | 상 |
| 0.3mm ~ 0.2mm | 중 |
| < 0.2mm | 하 |

</br></br>
### 3. 균열 위치 리포트
균열이 탐지 되었을 때의 드론 GPS 정보를 리포트합니다.

</br>
</br>

## 사용 기술

### 1. 균열 탐지
- Single Shot Multibox Detector(https://github.com/pierluigiferrari/ssd_keras) 를 사용하였습니다. 다른 Image Detection 딥러닝 알고리즘에 비해 속도와 탐지 성능 면에서 가장 바람직한 trade-off 관계를 보입니다. 즉, 속도는 빠르면서 높은 수준의 탐지 성능을 보입니다. Faster-RCNN 보다는 빠르며 YOLO 보다는 정확합니다. </br>
여기에 다양한 균열 이미지 데이터를 학습시켰습니다. Utah 주립 대학교에서 수집한 교량, 도보, 벽의 균열 이미지와, 중동 공과대학교(METU)에서 수집한 학교 건물의 균열 이미지를 사용하였습니다. </br>
<img src="https://user-images.githubusercontent.com/28426269/46650652-4bce8400-cbd8-11e8-8fc7-47dc05a67801.JPG" width="40%">

<사진 4. Utah 대학교에서 수집한 균열 이미지 데이터 세트>
<br> <br> 
<img src="https://user-images.githubusercontent.com/28426269/46650653-4c671a80-cbd8-11e8-8908-46d2d392d5a7.JPG" width="40%">

<사진 5. METU에서 수집한 균열 이미지 데이터 세트>
<br> <br> 
- 이 데이터 세트들은 0.06mm의 미세균열부터 25mm에 이르는 대형 균열까지, 균열 폭의 스펙트럼이 매우 넓어 실제 안전 진단 현장에서 발견할 수 있는 모든 종류의 균열을 반영합니다. 이처럼 다양한 크기, 텍스쳐, 노이즈를 반영한 총 1만여장의 균열 이미지 데이터를 학습하였습니다.
</br></br>
### 2. 균열 폭 측정

<img src="https://user-images.githubusercontent.com/28426269/46650969-6bb27780-cbd9-11e8-93b1-af089093aa66.JPG" width="40%">
<br>
<사진 6. 균열 폭 측정 작업 모식도>
<br><br>

* **Image Binarization**

이미지 내에서 균열인 부분과 균열이 아닌 부분을 분리하기 위해 이미지를 흑/백의 픽셀로 나누는 작업입니다.

* **Skeletonize**

균열의 중심 뼈대를 추출합니다. 균열의 진행 방향을 알 수 있습니다. Skeleton위의 픽셀에서 균열의 진행 방향에 수직선을 그으면 균열의 폭을 구할 수 있습니다.

* **Edge Detection**

균열의 외곽선을 추출합니다. Skeleton에서 균열의 진행 방향에 수직인 선과 균열의 외곽선이 함께 균열의 폭을 구하는 데 활용됩니다.
</br></br>

## Chamomile의 장점

### 1. 안전 진단의 자동화

콘크리트 구조물의 안전 진단을 일부 자동화 할 수 있습니다. Chamomile이 어떤 균열이 추가 진단이 필요할 지를 균열 폭에 근거하여 리포트합니다. 이는 후에 있을 추가 정밀 진단이 더욱 효율적으로 이루어지도록 돕습니다.

### 2. 비용 감소

구분 | 기존의 진단 방식 | 드론과 S/W를 통한 진단 방식
:---: | :---: | :---:
시간 | 많은 시간 소모<br>모든 부위를 굴절차에 탑승하여 직접 관찰하여 분석 | 짧고 정확한 분석<br>드론이 촬영<br>딥러닝 활용 정확한 자동 안전 진단
비용 | 10개 교량 기준 4천만원<br>굴절차 1회당 100여만원 비용<br>진단 인력, 분석 인력 비용 | 드론 촬영 1일 인건비 발생<br>초기 비용 : 드론 구입비 발생
인력 | 총 11명 소요<br>굴절차 운전 인원 1명<br>육안 점검 인원 2명<br>신호 통제 인원 8명 | 총 1~2명 소요<br>드론 조종 인원 및 조수 포함 


### 3. 안전 진단 시간 절약

추후 정밀 진단이 필요한 균열들의 위치와 심각한 정도를 사전에 알 수 있습니다. Chamomile 균열의 폭을 바탕으로 심각도가 높은 균열들부터 우선적으로 리포트하기 때문입니다. 따라서 안전 진단 시 소요 시간이 단축됩니다. 

### 4. 장기적으로 더욱 높은 수준의 안전을 담보 

딥러닝 균열 탐지 엔진은 전통적인 이미지 인식 알고리즘에 비해 상당히 높은 수준의 탐지율을 보입니다. 뿐만 아니라, 추가적인 데이터 수집을 통해 더 많고 다양한 데이터를 학습한다면 콘크리트 구조물의 텍스쳐에 성능이 영향을 받지 않게됩니다.

### 5. 범용성

콘크리트 균열은 건물의 종류, 지역, 국가와 상관없이 모두 비슷한 패턴을 보입니다. 즉, 어느 지역, 어느 종류의 대상에 적용이 되건 동일한 성능을 낼 수 있고, 별도의 수정없이 사용이 가능합니다.
</br>
</br>

## 성능


데이터 세트 | 탐지성능
:---: | :---:
METU data set | 85.7% (1714/2000)
실제 건물 외벽 촬영본 | 81.1% (43/53)
무작위로 수집한 균열 이미지 | 76% (19/25)

<br>
<br>

## 시스템 구성도

<img src="https://user-images.githubusercontent.com/28426269/46669839-869ddf80-cc0b-11e8-9f02-97b93b76fb20.png" width="85%">

## Framework
* **Crack detection** : Keras 2.2, Tensorflow-gpu 1.9.0, Python 3.6.6
* **Crack width estimation** : Scikit-image 0.14.0, Python 3.6.6
* **Web**
  -  UI : html, css, javascript, jQuery
  -  Web Server : Node.js, PythonShell
  -  Cloud Platform : Google Cloud Platform

</br></br>

## UI
<img src="https://user-images.githubusercontent.com/37619253/46622342-32d8bb00-cb65-11e8-8bf4-2d6e6668974d.png" width="100%">
</br>
1.  드론 영상 부분</br>
2.  영상 플레이 리스트(영상 섬네일 우측 파일 이름 클릭시 영상 실행)</br>
3.  크랙이 감지된 이미지</br>
4.  크랙 정보 리스트</br>


* 실행 동영상 : https://youtu.be/Yq5ULmGD-lw

* 해당 페이지 기능
1.  영상 옆 링크를 누르면 입력 된 영상 재생</br>
2.  영상을 균열 탐지 및 분석 엔진에 입력하고 코드를 실행</br>
3.  탐지된 균열 이미지 혹은 분석된 균열 정보를 클릭시 영상의 해당 play time으로 이동</br>
4.  탐지된 영상 시간, 균열의 폭, 위험군, GPS 좌표를 리포트. 균열의 폭이 0.3mm이상인 경우 (Risk가 '상'인경우) 빨간 글씨로 표시</br>
</br>
</br></br>

## 사용하는 방법
### 설치
* 균열 탐지 딥러닝

Anaconda, CUDA & CuDNN, Python, Tensorflow-gpu, Keras를 차례로 설치해야 합니다. 가상 환경을 만드는 복잡한 과정을 거치기 때문에, 과정을 상세히 설명한 링크를 첨부합니다. (https://medium.com/@viveksingh.heritage/how-to-install-tensorflow-gpu-version-with-jupyter-windows-10-in-8-easy-steps-8797547028a4)</br>
* 균열 폭 측정 알고리즘

Scikit-image 라이브러리를 사용합니다. 다음 명령어를 입력하여 설치합니다.</br>```pip install -U scikit-image```

### 데이터
* 균열 이미지 데이터
  -  METU 캠퍼스 균열 이미지 데이터 세트 : https://doi.org/10.15142/T3TD19
  -  Utah 대학교 균열 이미지 데이터 세트 : https://doi.org/10.15142/T3TD19
* 학습된 weight 파일 : ssd300_pascal_07+12_epoch-08_loss-1.9471_val_loss-1.9156.zip 파일을 다운로드 하여 압축 해제
* Annotation file : Annotation.zip 파일을 다운로드 하여 압축 해제.
* Annotation 툴 : 이미지 안에서 균열이 위치한 곳을 사람이 표시하여 저장하고, 이를 학습에 활용하기 위한 툴 https://github.com/tzutalin/labelImg 에서 다운로드하여 사용 가능

<br>

## 개발 문서

* Crack Width Estimation with Crack Detection v1.3.ipynb : 균열 인식 및 균열 폭 측정을 하는 파일입니다. 이 파일이 Chamomile에 사용되었습니다.
* SSD Crack Detection Training v1.3.ipynb : 균열 이미지 학습에 사용한 파일입니다
* SSD Crack Detection Inference v1.1.ipynb : 균열 이미지를 한 장씩 딥러닝 엔진에 입력하여 탐지 결과를 볼 수 있는 파일입니다.
* SSD Crack Detection Evaluation v1.0.ipynb : 균열 이미지 여러 장을 한꺼번에 입력하여 탐지율을 % 단위로 불 수 있는 파일입니다.
* crack_v5.py :  Crack Width Estimation with Crack Detection v1.3.ipynb 파일의 코드를 바탕으로, 서버에 업로드 하여 사용한 파일입니다.

위 파일들은 모두 일부 경로 변수를 수정하여 직접 사용이 가능합니다. 코드 내 주석을 통해 해당 내용을 설명하였습니다.
</br></br>

## 면담을 받은 공공기관
* 한국건설기술연구원
* 한국시설안전공단
* 한국시설안전공단 시설성능연구소
* 한국도로공사 도로교통연구원
* 경기도건설본부 도로건설과 도로시설팀

<br><br>

## 도움을 주신 분
* 한국도로공사 도로교통연구원 이병주 연구원님
* 서울시립대학교 토목공학과 조수진 교수님
* 경기도건설본부 도로건설과 도로시설팀장님

<br><br>

## Reference
1. Liu, Wei, et al. "SSD: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016. (Link : https://arxiv.org/abs/1512.02325) </br>
2. Kim, Hyunjun, et al. "Concrete crack identification using a UAV incorporating hybrid image processing." Sensors 17.9 (2017): 2052. (Link : http://www.mdpi.com/1424-8220/17/9/2052/htm)  </br>
3. Maguire, Marc; Dorafshan, Sattar; and Thomas, Robert J., "SDNET2018: A concrete crack image dataset for machine learning applications" (2018). Browse all Datasets. Paper 48. doi: https://doi.org/10.15142/T3TD19  </br>
4. Özgenel, Çağlar Fırat (2018), “Concrete Crack Images for Classification”, Mendeley Data, v1. doi: http://dx.doi.org/10.17632/5y9wdsg2zt.1  
 
</br></br>

## LICENSE
* 프로젝트 라이선스 : GPL-3.0
* Crack Width Estimation with Crack Detection v1.3.ipynb, crack_v5.py  :  Apache-2.0, BSD 3-Clause

</br>


# Concrete Crack Detection Using Drone & Deep Learning

</br>

## Introduction

Crack on the surface of concrete is the one of the most clear signs of deterioration of concrete structure. Therefore, the concrete crack on the surface is the first target for the safety inspection, in most cases. Since the concrete crack has typical patterns, software can support the structual health monitoring through automatic crack detection. However, there has been mainly hardware-supported approach to safety inspection, not software-based one. Software-supported approach can save the cost, time and effort for saftey inspection through automatic evaluation on concrete image data.
</br></br>
Plus, UAV(Unmanned Aerial Vehicle), or drone, can amplify the synergy for software-based safety inspection because it can easily approach where human is impossible to reach. Especially, drone becomes particularly useful if used to examine the large scale concrete structure where safety inspectors are not able to reach every parts directly. That's why drone is recently being actively researched and utilized for large scale concrete structure health monitoring system. Therefore, the software tool for drone-based safety inspection is needed for the efficient examination in the future.
</br></br>
However, there isn't open source software yet to detect crack in the video recorded by drone which shoots the concrete structure surface. Hereby, I share my own concrete crack detection software for drone-based safety inspection. Single Shot Multibox Detector(SSD) is used for detecting cracks and Hybrid image processing is employed to estimate the crack width.

</br>

## Problem of Current Safety Inspection Methods

The Current problem of hardware-based concrete structure safety inspection.

* Non-automation
</br>The assessment on data is still done by safety inspectors, even though the process can be automated through software because crack has typical patterns. Hardware can only help to collect data, but cannot make any decision on data instead of human.

* Expensive 
  -  The rental charge for under-bridge inspection vehicle : under-bridge inspection vehicle is used when inspecting the status of bridges. The problem is that the rental charges are expensive. It costs approximately 1,000 dollars(1,000,000 won) per one rental. The drone, however, demands only one purchase cost.<br>
  -  Additional manpower for inspection : when inspecting bridges, normally 8 people are needed to control the traffic, 2 people to get on the arm of the vehicle to examine the under-bridge, and 1 person to drive the vehicle. Totally, 11 people are required to inspect one bridge, which means not only rental cost for the special vehicle, but also additional employees are needed when using the existing way of structual health monitoring. Only 1 person will be needed if using drone for same purpose.

</br>

## The Objective of Chamomile
* Crack detection : Chamomile detects the cracks on the surface of the concrete structure.</br>
* Crack width estimation & classification based on seriousness: Chamomile estimate the width of the detected cracks, and reports the more serious crack first which has wider width than others. Basically, the crack of which width is more than 0.3mm is to be classified as "high risk crack", 0.3mm ~ 0.2mm as "low risk crack", and ~ 0.2mm as "minor crack.</br>
* Crack location reporting : Chamomile reports the actual location of the crack based on the flight log saved in the drone. With combining pixel information and the flight log, the location of crack can be calculated. It is useful for safety inspectors to know where the serious cracks locates which needs further precision diagnosis, before they physically approach to the target structure.

</br>
</br>

## User Benefit

* Automation of Safety Inspection

The examination on structual health can be partially automated. Chamomile selects which crack must be inspected based on width, which makes the further safety inspection done by human more efficient. Chamomile "filters".

* Reducing the time spent on Safety Inspection

* One-stop System 

* Assuring the higher level of safety in the long term

</br></br>

## Performance

Data set | Detection rate
:---: | :---:
METU data set | 85.7% (1714/2000)
Real concrete surface video | 81.1% (43/53)
Random crack images | 76% (19/25)


</br>

## Framework
* Crack detection : Keras 2.2, Tensorflow 1.9.0, Python 3.6.6</br>
* Crack width estimation : Scikit-image 0.14.0, Python 3.6.6</br>


</br>

## Special Thanks to
* Soojin Cho, Assistant Professor of Department of Civil Engineering, University of Seoul.

</br>

## Reference
1. Liu, Wei, et al. "SSD: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016. (Link : https://arxiv.org/abs/1512.02325) </br>
2. Kim, Hyunjun, et al. "Concrete crack identification using a UAV incorporating hybrid image processing." Sensors 17.9 (2017): 2052. (Link : http://www.mdpi.com/1424-8220/17/9/2052/htm)
