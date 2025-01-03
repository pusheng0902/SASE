This repository contains the code implementation of our paper: SASE: A Searching Architecture for Squeeze and Excitation Operations
  
Codes for SASE searching on CIFAR-10 and training & validation on ImageNet-1k are contained in dir: SASE_search  
Codes for SASE training & validation on COCO are contained in dir: mmdetection

For mmdetection,
1. Codes for implementation of SASE are located at: mmdetection/mmdet/models/backbones/resnet.py  

2. Codes for operations and operation sets are located at: mmdetection/mmdet/models/backbones/operations.py  

3. Weight files can be downloaded from: https://drive.google.com/file/d/18Varla-lIlsfn7hofsgn55fvLFALaUqf/view?usp=sharing  
    Please move them to mmdetection/weights.  
	Pretrained weights: pnorm_imgnet_r50.pth , pnorm_imgnet_r101.pth  
	Weights of the resulting architecture for detection: pnorm_det_r50_ep12.pth , pnorm_det_r101_ep12.pth ,  
	Weights of the resulting architecture for segmentation: pnorm_seg_r50_ep12.pth , pnorm_seg_r101_ep12.pth

4. To test on COCO 2017val with the resulting architecture:  
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