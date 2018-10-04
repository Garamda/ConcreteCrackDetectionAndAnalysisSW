# 드론을 활용한 콘크리트 구조물 균열 탐지 소프트웨어

</br>

## 소개
콘크리트 구조물을 안전 진단 할 때, 전문가들이 가장 먼저 보는 것이 외관 상태입니다. 그 중에서도 외벽상의 균열은 가장 확실한 상태판단 요소입니다. 때문에 안전 진단 시 가장 먼저 살피는 이상 징후 중 하나가 균열입니다. 콘크리트 균열은 일정한 패턴을 가지고 있으므로, 안전 진단 시 소프트웨어를 통한 균열 탐지 자동화가 가능합니다. 하지만 아직까지는 주로 특수 장비를 통해서만 진단을 해 왔을 뿐입니다. 소프트웨어를 통한 콘크리트 구조물의 균열 탐지는 안전 진단에 들어가는 비용, 시간, 노력을 크게 줄여줍니다.
</br></br>
뿐만 아니라, 드론이 여기에 함께 사용된다면 더욱 효율적인 안전 진단이 가능합니다. 드론은 사람이 직접 닿기 어려운 부분에 쉽게 접근할 수 있기 때문입니다. 특히 대형 콘크리트 구조물을 진단할 때 드론은 더욱 유용하게 쓰입니다. 대형 구조물은 안전 진단 인력이 모든 부분을 직접 살피기 불가능하기 때문입니다. 이러한 이유로 드론은 최근들어 대형 콘크리트 구조물의 안전 진단을 위해 활발히 연구 및 활용되고 있습니다. 따라서 드론을 활용한 안전 진단에 맞는 SW가 필요합니다.
</br></br>
하지만 드론으로 촬영한 콘크리트 외벽 영상에서 균열을 탐지하고 분석하는 오픈소스 SW는 아직 없는 실정입니다. 이제 SPARK에서 드론을 활용한 안전 진단을 위한 콘크리트 균열 탐지 SW를 사용하실 수 있습니다. 균열을 탐지하기 위한 딥러닝 알고리즘으로는 Single Shot Multibox Detector(https://github.com/pierluigiferrari/ssd_keras)를 사용했습니다. 균열의 폭을 측정하기 위해서 Image binarization, Skeletonize, Edge detection의 전처리 방법론을 사용하였습니다.

</br>

## 기존 안전 진단의 문제점
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
