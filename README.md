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
## The Role of SPARK (role? 이상함)
1. Crack detection
2. Crack width estimation
3. Crack location reporting
</br>
## User Benefit
1. 
2. 
3. 
4. 
</br>
## Performance
도표 사용, 실제 test 결과를 넣을 것(코드 사용하여 evaluation한 결과)
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

</br>
## Development Documentation

</br>
## Reference
(링크 달아주기)</br>
1. SSD 논문 </br>
2. 균열 폭 논문

## LICENSE
</br>
