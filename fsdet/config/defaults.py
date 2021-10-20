from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True

_CC.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT = 1.0

_CC.MODEL.ROI_BOX_HEAD.SOFT_LABEL_BRANCH = CN()
_CC.MODEL.ROI_BOX_HEAD.SOFT_LABEL_BRANCH.TEMPERATURE = 5
_CC.MODEL.ROI_BOX_HEAD.SOFT_LABEL_BRANCH.LOSS_WEIGHT = 0.2
_CC.MODEL.ROI_BOX_HEAD.SOFT_LABEL_BRANCH.BASE_WEIGHTS = "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth"
_CC.MODEL.ROI_BOX_HEAD.SOFT_LABEL_BRANCH.HEAD_ONLY = False
_CC.MODEL.ROI_BOX_HEAD.SOFT_LABEL_BRANCH.NUM_CLASSES = 60
