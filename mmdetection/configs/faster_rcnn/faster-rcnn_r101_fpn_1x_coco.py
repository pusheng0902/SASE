_base_ = './faster-rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=None))

load_from = 'weights/pnorm_imgnet_r101.pth'