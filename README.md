# ESGAN-tensorflow
A simple tensorflow implementation of ESGAN that can transfer image emotion
## Requirement
* Tensorflow 1.9
* python 3.6
* scipy
## Usage
### Train
* bash run_train.sh
### Test
* bash run_test.sh
## Method Overview
![image](https://github.com/canqChen/ESGAN/blob/master/images/method%20overview.png)
## Architecture
![image](https://github.com/canqChen/ESGAN/blob/master/images/model_detail.png)
## Results
### Sample Examples
![image](https://github.com/canqChen/ESGAN/blob/master/images/joy2sadness.jpg "joy2sadness")
![image](https://github.com/canqChen/ESGAN/blob/master/images/sadness2joy.jpg "sadness2joy")
![image](https://github.com/canqChen/ESGAN/blob/master/images/joy2anger.jpg "joy2anger")
![image](https://github.com/canqChen/ESGAN/blob/master/images/anger2joy.jpg "anger2joy")
![image](https://github.com/canqChen/ESGAN/blob/master/images/surprise2sadness.jpg "surprise2sadness")
![image](https://github.com/canqChen/ESGAN/blob/master/images/sadness2surprise.jpg "sadness2surprise")
![image](https://github.com/canqChen/ESGAN/blob/master/images/surprise2anger.jpg "surprise2anger")
![image](https://github.com/canqChen/ESGAN/blob/master/images/anger2surprise.jpg "anger2surprise")
### Example-guided Examples
![image](https://github.com/canqChen/ESGAN/blob/master/images/eg_joy_sadness.jpg "Example-guided joy↔sadness")
![image](https://github.com/canqChen/ESGAN/blob/master/images/eg_joy_anger.jpg "Example-guided joy↔anger")
![image](https://github.com/canqChen/ESGAN/blob/master/images/eg_surprise_sadness.jpg "Example-guided surprise↔sadness")
![image](https://github.com/canqChen/ESGAN/blob/master/images/eg_surprise_anger.jpg "Example-guided surprise↔anger")
## Code Reference
* [CycleGAN-Tensorflow](https://github.com/taki0112/CycleGAN-Tensorflow)
