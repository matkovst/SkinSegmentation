# SkinSegmentation
## Description
A model for human skin segmentation on an image/video. Based on Bayesian probability estimation: 

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=P(skin|rgb)=\frac{P(rgb|skin)P(skin)}{P(rgb|skin)P(skin)&plus;P(rgb|\overline{skin})P(\overline{skin})}" target="_blank"><img src="https://latex.codecogs.com/png.latex?P(skin|rgb)=\frac{P(rgb|skin)P(skin)}{P(rgb|skin)P(skin)&plus;P(rgb|\overline{skin})P(\overline{skin})}" title="P(skin|rgb)=\frac{P(rgb|skin)P(skin)}{P(rgb|skin)P(skin)+P(rgb|\overline{skin})P(\overline{skin})}" /></a>
</p>
where skin likelihood and nonskin likelihood were estimated by GMM (GMM parameters were taken from this paper https://www.hpl.hp.com/techreports/Compaq-DEC/CRL-98-11.pdf).

The code can be launched both on CPU and GPU.


Original             |  Segmented
:-------------------------:|:-------------------------:
![](https://github.com/matkovst/SkinSegmentation/blob/master/data/orig.jpg)  |  ![](https://github.com/matkovst/SkinSegmentation/blob/master/data/result_gpu.jpg)


## Requirements
- OpenCV 4.1.2
- (*optional*) OpenMP 2.0
- (*optional*) CUDA 10.0

## Running
`SkinSegmentation.exe INPUT [RESIZE] [ROTATE]`
