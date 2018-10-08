# 드론을 활용한 콘크리트 구조물 균열 탐지 소프트웨어

</br>

## 소개
콘크리트 구조물을 안전 진단 할 때, 전문가들이 가장 먼저 보는 것이 외관 상태입니다. 그 중에서도 외벽상의 균열은 가장 확실한 상태판단 요소입니다. 때문에 안전 진단 시 가장 먼저 살피는 이상 징후 중 하나가 균열입니다. 
(진단 현장 사진 추가)
콘크리트 균열은 일정한 패턴을 가지고 있으므로, 안전 진단 시 소프트웨어를 통한 균열 탐지 자동화가 가능합니다. 하지만 아직까지는 주로 특수 장비를 통해서만 진단을 해 왔을 뿐입니다. 소프트웨어를 통한 콘크리트 구조물의 균열 탐지는 안전 진단에 들어가는 비용, 시간, 노력을 크게 줄여줍니다.
</br></br>
뿐만 아니라, 드론이 여기에 함께 사용된다면 더욱 효율적인 안전 진단이 가능합니다. 드론은 사람이 직접 닿기 어려운 부분에 쉽게 접근할 수 있기 때문입니다. 특히 대형 콘크리트 구조물을 진단할 때 드론은 더욱 유용하게 쓰입니다. 대형 구조물은 안전 진단 인력이 모든 부분을 직접 살피기 불가능하기 때문입니다. 이러한 이유로 드론은 최근들어 대형 콘크리트 구조물의 안전 진단을 위해 활발히 연구 및 활용되고 있습니다. (드론 활용 진단 사진 추가) 따라서 드론을 활용한 안전 진단에 맞는 SW가 필요합니다.
</br></br>
하지만 드론으로 촬영한 콘크리트 외벽 영상에서 균열을 탐지하고 분석하는 오픈소스 SW는 아직 없는 실정입니다. 이제 SPARK에서 드론을 활용한 안전 진단을 위한 콘크리트 균열 탐지 SW를 사용하실 수 있습니다. 균열을 탐지하기 위한 딥러닝 알고리즘으로는 Single Shot Multibox Detector(https://github.com/pierluigiferrari/ssd_keras) 를 사용했습니다. 균열의 폭을 측정하기 위해서 Image binarization, Skeletonize, Edge detection의 전처리 방법론을 사용하였습니다.

</br>

## 기존 안전 진단의 문제점

1. 자동화 되지 않은 진단
</br>데이터에 대한 모든 판단은 안전 진단 인력이 직접 해야합니다. 균열은 일정한 패턴을 가지고 있기 때문에 SW를 통한 자동화가 가능함에도, 특수 장비는 데이터를 수집하는 것만 도울 뿐입니다.

</br>

2. 높은 비용 
<img src="https://user-images.githubusercontent.com/28426269/46481857-3bf91d80-c82f-11e8-8a9f-718a18bb5e86.jpg" width="70%">

교량 안전 진단을 예시로 설명하겠습니다.

[1] 굴절차 대여 비용 : 교량 하부를 진단할 때에는 위 사진과 같이 굴절차를 사용합니다. 문제는 굴절차 1회 대여에 약 100만원 가량의 높은 비용이 발생한다는 점입니다. 드론과 SPARK를 사용하면 초기 구매 비용만 소요됩니다.<br></br>
[2] 투입 인력 : 교량 안전 진단 시 평균적으로 11명의 인력이 필요합니다. 굴절차를 운전하는 인원 1명, 굴절차의 팔에 탑승하여 교량 하부를 점검하는 인원 2명, 교량 상부에서 신호 통제를 하는 인원 8명이 필요합니다. 굴절차 대여 비용과 더불어 인력 고용의 비용까지 들어가는 것입니다. 이와 같은 이유로, 10개 교량을 기준으로 평균 4천만원의 진단 비용이 발생합니다. 드론과 SPARK를 사용한다면 드론을 운전하는 인력 1명만이 필요합니다.

</br>
</br>

## SPARK의 목표
1. 균열 탐지 : 콘크리트 외벽상의 균열을 탐지합니다.</br></br>
2. 균열 폭 측정 & 심각도에 따른 분류: 탐지된 균열의 폭을 측정하고, 폭이 큰 심각한 균열을 우선적으로 리포트 합니다. 균열의 폭을 측정하는 이유는 실제 안전 진단 현장에서 위험도를 판단하기 위해 사용하는 중요한 정보이기 때문입니다. 폭이 0.3mm보다 크면 유지 보수 작업에 들어가고, 0.3mm~0.2mm에 해당하면 추이를 지켜봅니다. 0.2mm 미만인 균열은 위험도가 낮다고 판단합니다. SPARK에서는 이 기준을 그대로 사용하였습니다. 폭이 0.3mm보다 큰 균열은 "상"위험군으로, 0.3mm ~ 0.2mm의 균열은 "중"위험군으로, 0.2mm 미만의 균열은 "하" 위험군으로 분류합니다.</br></br>
3. 균열 위치 리포트 : 균열이 탐지 되었을 때의 드론 GPS 정보를 리포트합니다.

</br>
</br>

## 사용 기술
1. 균열 탐지 : Single Shot Multibox Detector(https://github.com/pierluigiferrari/ssd_keras) 를 사용하였습니다. 다른 Image Detection 딥러닝 알고리즘에 비해 속도와 탐지 성능 면에서 가장 바람직한 trade-off 관계를 보입니다. 즉, 속도는 빠르면서 높은 수준의 탐지 성능을 보입니다. Faster-RCNN 보다는 빠르며 YOLO 보다는 정확합니다. </br>
여기에 다양한 균열 이미지 데이터를 학습시켰습니다. Utah 주립 대학교에서 수집한 교량, 도보, 벽의 균열 이미지와, 중동 공과대학교(METU)에서 수집한 학교 건물의 균열 이미지를 사용하였습니다. 이 데이터 세트들은 0.06mm의 미세균열부터 25mm에 이르는 대형 균열까지, 균열 폭의 스펙트럼이 매우 넓어 실제 안전 진단 현장에서 발견할 수 있는 모든 종류의 균열을 반영합니다. 이처럼 다양한 크기, 텍스쳐, 노이즈를 반영한 총 15000여장의 균열 이미지 데이터를 학습하였습니다.
</br></br>
2. 균열 폭 측정
</br>
1) Image Binarization
</br>
이미지 내에서 균열인 부분과 균열이 아닌 부분을 분리하기 위해 이미지를 흑/백의 픽셀로 나누는 작업입니다.
2) Skeletonize
</br>
균열의 중심 뼈대를 추출합니다. 균열의 진행 방향을 알 수 있어, 균열의 폭을 구하는 데 활용됩니다.
</br>
3) Edge Detection
</br>
균열의 외곽선을 추출합니다. 균열의 폭을 구하는 데 활용됩니다.

## SPARK의 장점
1. 안전 진단의 자동화
</br>
콘크리트 구조물의 안전 진단을 일부 자동화 할 수 있습니다. SPARK가 어떤 균열이 추가 진단이 필요할 지를 균열 폭에 근거하여 리포트합니다. 이는 후에 있을 추가 정밀 진단이 더욱 효율적으로 이루어지도록 돕습니다.
</br></br>
2. 비용 감소
</br>

구분 | 기존의 진단 방식 | 드론과 S/W를 통한 진단 방식
:---: | :---: | :---:
시간 | 많은 시간 소모<br>모든 부위를 굴절차에 탑승하여 직접 관찰하여 분석 | 짧고 정확한 분석<br>드론이 촬영<br>딥러닝 활용 정확한 자동 안전 진단
비용 | 10개 교량 기준 4천만원<br>굴절차 1회당 100여만원 비용<br>진단 인력, 분석 인력 비용 | 드론 촬영 1일 인건비 발생<br>초기 비용 : 드론 구입비 발생
인력 | 총 11명 소요<br>굴절차 운전 인원 1명<br>육안 점검 인원 2명<br>신호 통제 인원 8명 | 총 1~2명 소요<br>드론 조종 인원 및 조수 포함 


</br></br>
3. 안전 진단 시간 절약
</br>
추후 정밀 진단이 필요한 균열들의 위치와 심각한 정도를 사전에 알 수 있습니다. SPARK가 균열의 폭을 바탕으로 심각도가 높은 균열들부터 우선적으로 리포트하기 때문입니다. 따라서 안전 진단 시 소요 시간이 단축됩니다. 
</br></br>
4. 장기적으로 더욱 높은 수준의 안전을 담보 
</br>
딥러닝 균열 탐지 엔진은 상당히 높은 수준의 탐지율을 보입니다. 
</br></br>
5. 범용성
</br>
콘크리트 균열은 건물의 종류, 지역, 국가와 상관없이 모두 비슷한 패턴을 보입니다. 즉, 어느 종류의 대상에 적용이 되건 상관없이 
</br></br>

</br></br>

## 성능
도표 사용, 실제 test 결과를 넣을 것(코드 사용하여 evaluation한 결과)

<br><br>

SPARK | Detection | ()
:---: | :---: | :---:
Crack Images from Google | d | d
d | d | d
d | d | d
d | d | d

<br><br>

SPARK | Width Estimation | ()
:---: | :---: | :---:
d | d | d
d | d | d
d | d | d

</br>

## Framework
1. Crack detection : Keras 2.2, Tensorflow-gpu 1.9.0, Python 3.6.6</br>
2. Crack width estimation : Scikit-image 0.14.0, Python 3.6.6</br>

</br></br>

## UI
<img src="https://user-images.githubusercontent.com/37619253/46622342-32d8bb00-cb65-11e8-8bf4-2d6e6668974d.png" width="100%">
</br>
1. 드론 영상 부분</br>
2. 영상 플레이 리스트(영상 섬네일 우측 파일 이름 클릭시 영상 실행)</br>
3. 크랙이 감지된 이미지</br>
4. 크랙 정보 리스트</br>
</br>
</br>
http://35.221.191.213:3000</br>
</br>
Front-End: html, css, JQuery</br>
Back-End: Node.js, Python-shell</br>
server: GCP</br>
</br>
해당 페이지 기능</br>
1. 영상 옆 링크 누르면 보딩 된 영상 재생</br>
2. 영상이 재생 되면서 크랙이 감지된 경우 가운데와 오른쪽에 이미지와 크랙정보 생성</br>
3. 가운데 이미지 혹은 크랙 정보를 클릭시 해당 영상의 위치로 이동</br>
4. 크랙의 균열이 0.0~0.2: 하0.2~0.3: 중0.3mm이상인 경우(Risk가 '상'인경우) 빨간 글씨로 표시(나머지는 검은글씨)</br>
추후 정밀 진단이 필요할 경우 실제 위치를 알기 위한 촬영 당시의 GPS좌표 출력</br>
</br></br>
## 사용하는 방법
1. 균열 탐지 딥러닝 : Anaconda, CUDA & CuDNN, Python, Tensorflow-gpu, Keras를 차례로 설치해야 합니다. 가상 환경을 만드는 복잡한 과정을 거치기 때문에, 과정을 상세히 설명한 링크를 첨부합니다. (https://medium.com/@viveksingh.heritage/how-to-install-tensorflow-gpu-version-with-jupyter-windows-10-in-8-easy-steps-8797547028a4)</br>
2. 균열 폭 측정 알고리즘 : Scikit-image 라이브러리를 사용합니다. 다음 명령어를 입력하여 설치합니다.</br>```pip install -U scikit-image```
</br></br>
1. 균열 이미지 데이터</br>
1) METU 캠퍼스 균열 이미지 데이터 세트 :  https://data.mendeley.com/datasets/5y9wdsg2zt/1 </br>
2) </br>
2. 학습된 weight 파일 : </br>
3. 크랙 이미지 데이터 링크, h5 파일 링크, annotation file들 링크도 -> 구글 드라이브 링크로 </br>
4. Annotation 툴 : https://github.com/tzutalin/labelImg </br>
기본 설정에 기반한 내 친절한 설명이 들어가 있어야 함 -> 내 오픈소스 장점 어필 가능</br>
</br>

## 개발 문서
</br>
파일들간 관계를 설명, 중요한 파일들 위주로 설명 13 ~ 15 참조, EX) 얘는 학습용, 얘는 진단용, 얘는 UI, 얘는 라이브러리 등등 큰 범주들을 쓴 후 각 부분들을 들어가서 자세히, 폴더 구조로 해보자! 이게 제일 좋을듯 / 설정 파일(유저가 직접 변경해서 사용해야 하는 경우)은 
</br></br>

## 면담을 받은 공공기관
1. 한국건설기술연구원</br>
2. 한국시설안전공단</br>
3. 한국시설안전공단 시설성능연구소</br>
4. 한국도로공사 도로교통연구원</br>
5. 경기도건설본부 도로건설과 도로시설팀</br>

<br><br>

## Reference
1. Liu, Wei, et al. "SSD: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016. (Link : https://arxiv.org/abs/1512.02325) </br>
2. Kim, Hyunjun, et al. "Concrete crack identification using a UAV incorporating hybrid image processing." Sensors 17.9 (2017): 2052. (Link : http://www.mdpi.com/1424-8220/17/9/2052/htm)

</br></br>

## LICENSE
오픈소스 어느부분 수정했는지 아주 큰 범주로만 

</br></br></br>

# Concrete Crack Detection Using Drone & Deep Learning

</br>

## Introduction
주석 싹다 정리!!!
Crack on the surface of concrete is the one of the most clear signs of deterioration of concrete structure. Therefore, the concrete crack on the surface is the first target for the safety inspection, in most cases. Since the concrete crack has typical patterns, software can support the structual health monitoring through automatic crack detection. However, there has been mainly hardware-supported approach to safety inspection, not software-based one. Software-supported approach can save the cost, time and effort for saftey inspection through automatic evaluation on concrete image data.
</br></br>
Plus, UAV(Unmanned Aerial Vehicle), or drone, can amplify the synergy for software-based safety inspection because it can easily approach where human is impossible to reach. Especially, drone becomes particularly useful if used to examine the large scale concrete structure where safety inspectors are not able to reach every parts directly. That's why drone is recently being actively researched and utilized for large scale concrete structure health monitoring system. Therefore, the software tool for drone-based safety inspection is needed for the efficient examination in the future.
</br></br>
However, there isn't open source software yet to detect crack in the video recorded by drone which shoots the concrete structure surface. Hereby, I share my own concrete crack detection software for drone-based safety inspection. Single Shot Multibox Detector(SSD) is used for detecting cracks and Hybrid image processing is employed to estimate the crack width.

</br>

## Problem of Current Safety Inspection Methods
The Current problem of hardware-based concrete structure safety inspection.
the Korea Expressway Corporation(한국도로공사) U-BIROS
https://github.com/Garamda/SPARK/blob/master/images/U-BIROS.jpg

1. Automatation
</br>The assessment on data is still done by safety inspectors, even though the process can be automated through software because crack has typical patterns. Hardware can only help to collect data, but cannot make any decision on data instead of human.

2. Expensive 
1) The rental charge for under-bridge inspection vehicle : under-bridge inspection vehicle is used when inspecting the status of bridges. The problem is that the rental charges are expensive. It costs approximately 1,000 dollars(1,000,000 won) per one rental. The drone, however, demands only one purchase cost.<br>
2) Additional manpower for inspection : when inspecting bridges, normally 8 people are needed to control the traffic, 2 people to get on the arm of the vehicle to examine the under-bridge, and 1 person to drive the vehicle. Totally, 11 people are required to inspect one bridge, which means not only rental cost for the special vehicle, but also additional employees are needed when using the existing way of structual health monitoring. Only 1 person will be needed if using drone for same purpose.

</br>

## The Objective of SPARK
1. Crack detection : SPARK detects the cracks on the surface of the concrete structure.</br>
2. Crack width estimation & classification based on seriousness: SPARK estimate the width of the detected cracks, and reports the more serious crack first which has wider width than others. Basically, the crack of which width is more than 0.3mm is to be classified as "high risk crack", 0.3mm ~ 0.2mm as "low risk crack", and ~ 0.2mm as "minor crack.</br>
3. Crack location reporting : SPARK reports the actual location of the crack based on the flight log saved in the drone. With combining pixel information and the flight log, the location of crack can be calculated. It is useful for safety inspectors to know where the serious cracks locates which needs further precision diagnosis, before they physically approach to the target structure.

</br>
</br>

## User Benefit

1. Automation of Safety Inspection
</br>
The examination on structual health can be partially automated. SPARK selects which crack must be inspected based on width, which makes the further safety inspection done by human more efficient. SPARK "filters".
</br></br>
2. Reducing the time spent on Safety Inspection
</br>
추후 정밀 진단이 필요한 균열들의 위치와 심각한 정도를 사전에 알 수 있습니다. SPARK가 균열의 폭을 바탕으로 심각도가 높은 균열들부터 우선적으로 리포트하기 때문입니다. 따라서 안전 진단 시 소요 시간이 단축됩니다. 
</br></br>
3. One-stop System 
</br>
드론을 통한 데이터 촬영과 균열 검출 엔진을 연동 -> 실시간 처리 가능
구조물을 촬영부터 S/W를 통한 분석 작업 : 중단 없는 하나의 프로세스
</br>
4. Assuring the higher level of safety in the long term

</br></br>

## Performance
도표 사용, 실제 test 결과를 넣을 것(코드 사용하여 evaluation한 결과), 업데이트

<br><br>

SPARK | Detection | ()
:---: | :---: | :---:
Crack Images from Google | d | d
d | d | d
d | d | d
d | d | d

<br><br>

SPARK | Width Estimation | ()
:---: | :---: | :---:
d | d | d
d | d | d
d | d | d

</br>

## Framework
1. Crack detection : Keras 2.2, Tensorflow 1.9.0, Python 3.6.6</br>
2. Crack width estimation : Scikit-image 0.14.0, Python 3.6.6</br>
3. Crack location reporting : </br>

</br>

## How to use
1. Crack detection : Install Anaconda, Keras, Tensorflow, Python -> 명령어 써주기
2. Crack width estimation : Install Scikit-image -> pip install -U scikit-image </br>
3. Crack location reporting : </br>
크랙 이미지 데이터 링크, h5 파일 링크, annotation file들 링크도 -> 구글 드라이브 링크로 </br>
기본 설정에 기반한 내 친절한 설명이 들어가 있어야 함 -> 내 오픈소스 장점 어필 가능</br>
</br>

## Development Documentation
</br>
파일들간 관계를 설명, 중요한 파일들 위주로 설명 13 ~ 15 참조, EX) 얘는 학습용, 얘는 진단용, 얘는 UI, 얘는 라이브러리 등등 큰 범주들을 쓴 후 각 부분들을 들어가서 자세히, 폴더 구조로 해보자! 이게 제일 좋을듯 / 설정 파일(유저가 직접 변경해서 사용해야 하는 경우)은 
</br></br>

## Consultation
면담 결과를 표로 넣을것(객관적인 근거)

<br><br>

ㅇㅁㅇ | 대상 교량 | 시행 횟수 | 점검 수준 | 굴절차 필요 여부 | 비용(교량의 종류 및 길이에 따라 금액 산정이 달라지기는 하지만, 평균적으로)
| 필요 인원 | 소요 시간 |
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
정기점검 | d | d | d | d | d | d | d |
정밀 안전점검 | d | d | d | d | d | d | d |
정밀 진단 | d | d | d | d | d | d | d |

</br>
2.	드론을 활용하여 균열을 탐지하고 균열의 폭과 위치를 알려주는 소프트웨어를 사용한다면, 교량을 정기점검 할 때 인력, 시간, 비용을 어느 정도로 절감할 수 있을까요?
1)	굴절차는 1회 대여에 대략 100만원 이상이 필요합니다. 뿐만 아니라 사용시에 교량 상부에서 따로 신호통제를 필요로 하기 때문에, 8명 정도의 추가 인력이 필요합니다. 드론을 사용하면 굴절차 대여, 신호통제의 문제가 생기지 않기 때문에, 이런 면에서는 드론이 장점을 가진다고 할 수 있습니다.
2)	세 가지 종류의 안전 진단 중, 육안 점검만을 실시하는 정기점검을 커버할 수 있지 않을까 생각합니다. 다만 정기점검시에도 균열뿐만 아니라 다양한 이상 징후들 역시 관찰하므로, 균열만을 탐지하는 것은 제한적인 솔루션이 될 것입니다.
</br>
-	드론과 SW를 활용하면, 이상징후의 시간에 따른 내역이 데이터로 관리가 된다는 점이 좋습니다. 
</br>

</br>

## Reference
1. Liu, Wei, et al. "SSD: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016. (Link : https://arxiv.org/abs/1512.02325) </br>
2. Kim, Hyunjun, et al. "Concrete crack identification using a UAV incorporating hybrid image processing." Sensors 17.9 (2017): 2052. (Link : http://www.mdpi.com/1424-8220/17/9/2052/htm)

</br>

## LICENSE
오픈소스 어느부분 수정했는지 아주 큰 범주로만 
</br>
