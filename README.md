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
the Korea Expressway Corporation(한국도로공사) U-BIROS
https://github.com/Garamda/SPARK/blob/master/images/U-BIROS.jpg

1. Automatation : The assessment on data is still done by safety inspectors, even though the process can be automated through software because crack has typical patterns. Hardware can only help to collect data, but cannot make any decision on data instead of human.

2. Expensive : 면담 내용 채워넣기

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
-  데이터에 대한 판단 -> 자동화 
S/W가 1차적으로 위험 징후를 탐지, 추가 진단이 부분들을 선별
</br></br>
2. Reducing the time spent on Safety Inspection
</br>
추후 정밀 진단이 필요한 균열들의 위치를 미리 파악 
S/W가 리포트 한 부분들에 대해서만 정밀진단 -> 안전 진단 시 소요 시간이 단축
</br></br>
3. One-stop System 
</br>
드론을 통한 데이터 촬영과 균열 검출 엔진을 연동 -> 실시간 처리 가능
구조물을 촬영부터 S/W를 통한 분석 작업 : 중단 없는 하나의 프로세스
</br>
4. Assuring the higher level of safety in the long term

</br></br>

## Performance
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
1. Crack detection : Keras 2.2, Tensorflow 1.9.0, Python 3.6.6</br>
2. Crack width estimation : Scikit-image 0.14.0, Python 3.6.6</br>
3. Crack location reporting : </br>

</br>

## How to use
1. Crack detection : Install Anaconda, Keras, Tensorflow, Python -> 명령어 써주기
2. Crack width estimation : Install Scikit-image -> pip install -U scikit-image </br>
3. Crack location reporting : </br>
크랙 이미지 데이터 링크, h5 파일 링크, annotation file들 링크도 </br>
</br>

## Development Documentation

</br>

## Reference
1. Liu, Wei, et al. "SSD: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016. (Link : https://arxiv.org/abs/1512.02325) </br>
2. Kim, Hyunjun, et al. "Concrete crack identification using a UAV incorporating hybrid image processing." Sensors 17.9 (2017): 2052. (Link : http://www.mdpi.com/1424-8220/17/9/2052/htm)

</br>

## LICENSE
</br>
