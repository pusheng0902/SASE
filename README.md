--------------------------------------------------------------------------
Our environment:
OS: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-88-generic x86_64)
GPU: 40GB Tesla A100
CPU: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz
python: 3.9.2
CUDA: v11.3
Torch: 1.12.1+cu113
--------------------------------------------------------------------------
Codes for SASE searching on CIFAR-10 and training & validation on ImageNet-1k are contained in dir: SASE_search
Codes for SASE training & validation on COCO are contained in dir: mmdetection


For mmdetection,
1. Codes for implementation of SASE are located at: mmdetection/mmdet/models/backbones/resnet.py

2. Codes for operations and operation sets are located at: mmdetection/mmdet/models/backbones/operations.py

3. Weight files can be downloaded from: https://drive.google.com/file/d/18Varla-lIlsfn7hofsgn55fvLFALaUqf/view?usp=sharing
    Please move them to mmdetection/weights.
	Pretrained weights: pnorm_imgnet_r50.pth , pnorm_imgnet_r101.pth
	Resulting weights for detection: pnorm_det_r50_ep12.pth , pnorm_det_r101_ep12.pth , 
	.......................................for segmentation: pnorm_seg_r50_ep12.pth , pnorm_seg_r101_ep12.pth

4. To test on COCO 2017val with resulting weights:
	cd mmdetection/
	python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

5. To train on COCO2017train from pretrained weights:
	cd mmdetection/
	python tools/train.py ${CONFIG_FILE}


For SASE_search,
1. Code for operations and operation sets are located at: SASE_search/operations.py

2. Please run the following command to perform searching:
python train_search_attention.py

3. Please run the following command to perform training and validation on ImageNet:
python train_imagenet.py