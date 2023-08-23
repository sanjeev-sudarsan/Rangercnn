## Introduction
A range image based 3D object detector based on the paper RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation (https://arxiv.org/abs/2009.00206). The detector is single stage as I did not implement the two stage model yet.

## Performance

	|     Rangercnn     |      Easy     |   Moderate    |     Hard      |
	| ----------------- | ------------- | ------------- | ------------- |
	|   Car AP@0.70 3D  |     83.0985   |    73.4474    |    70.2410    |
	| Car AP_R40@0.70 3D|     84.3880   |    73.3036    |    71.1341    |


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.


## Quick Demo
Please refer to [DEMO.md](docs/DEMO.md) for a quick demo to test with a pretrained model and 
visualize the predicted results on your custom data or the original KITTI data.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

