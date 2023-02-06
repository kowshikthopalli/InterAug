
import warnings
import pycocotools.mask as mask_util
# Commented out IPython magic to ensure Python compatibility.
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
# import some common libraries
import numpy as np
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.model_zoo import model_zoo
# %matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from matplotlib.image import BboxImage
from PIL import Image
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
import torch
#from utils_aug import *

warnings.filterwarnings("ignore")

def Solarize(img, boxs, level):  # [0, 256]  
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    solarize = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.Solarize(level, threshold=(32, 128)))
    img_aug, bb_aug= solarize(image=img, bounding_boxes=bbs)
    return img_aug

def auto_contrast(img, boxs,level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape) 
    auto_cont = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.pillike.Autocontrast(level))
    img_aug, bb_aug = auto_cont(image=img, bounding_boxes=bbs)
    return img_aug

def emboss(img, boxs,level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    emboss = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.Emboss(alpha=(0, 1.0), strength=level))
    img_aug, bb_aug = emboss(image=img, bounding_boxes=bbs)
    #return img_aug, bb_aug
    return img_aug

def Enhancecolor(img, boxs,level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    enhancecolor = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.pillike.EnhanceColor(level))
    img_aug, bb_aug = enhancecolor(image=img, bounding_boxes=bbs)
    return img_aug

def EnhanceSharpness(img, boxs, level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    enhanceSharpness = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.pillike.EnhanceSharpness(level))
    img_aug, bb_aug = enhanceSharpness(image=img, bounding_boxes=bbs)
    return img_aug

def Posterize(img, boxs, level): 
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    posterize = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.pillike.Posterize(int(level)))
    img_aug, bb_aug = posterize(image=img, bounding_boxes=bbs)
    return img_aug

def Enhancecontrast(img, boxs, level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    enhancecontrast = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.pillike.EnhanceContrast(level))
    img_aug, bb_aug = enhancecontrast(image=img, bounding_boxes=bbs)
    return img_aug

def Brightness(img, boxs, level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    brightness = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.pillike.EnhanceBrightness(level))
    img_aug, bb_aug = brightness(image=img, bounding_boxes=bbs)
    return img_aug

def AddToHue(img, boxs, level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    addtohue = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.AddToHue((-255,255)))
    img_aug, bb_aug = addtohue(image=img, bounding_boxes=bbs)
    return img_aug

def Blur(img, boxs, level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    blur = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.OneOf([iaa.GaussianBlur(sigma =(0.0, 3.0)),iaa.MedianBlur(k=(3, 7)),iaa.MotionBlur(k=15)]))
    img_aug, bb_aug = blur(image=img, bounding_boxes=bbs)
    return img_aug

def Noise(img, boxs, level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    noise = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.OneOf([iaa.imgcorruptlike.GaussianNoise(severity=2),iaa.imgcorruptlike.ImpulseNoise(severity=2),iaa.imgcorruptlike.ShotNoise(severity=2)]))
    img_aug,bbs_aug = noise(image=img, bounding_boxes=bbs)
    return img_aug

def Clouds(img, boxs, level):
    bbs=BoundingBoxesOnImage([BoundingBox(*bb)for bb in boxs],shape=img.shape)
    cloudaug = iaa.BlendAlphaBoundingBoxes(None,foreground=iaa.BlendAlphaMask(iaa.InvertMaskGen(level, iaa.VerticalLinearGradientMaskGen()),iaa.Clouds()))
    img_aug,bbs_aug = cloudaug(image=img, bounding_boxes=bbs)
    return img_aug

def fliplr(img, boxes, level):
    flip = iaa.Fliplr(level)
    img_aug = flip(image=img)
    return img_aug

def centercrop(img, boxes, level):
    crop = iaa.CenterCropToAspectRatio(level)
    img_aug = crop(image=img)
    return img_aug


ALL_TRANSFORMS = [
            (fliplr, 0.5, 1.0),
            (centercrop, 1.0, 1.5),
            (auto_contrast, 10, 20),
            (Solarize, 0.2, 1.0),       
            (emboss, 0.5, 2.0),
            (Enhancecolor, 0.5, 3.0),
            (EnhanceSharpness, 0.5, 2.0),
            (Enhancecontrast, 0.5, 1.5),  
            (Posterize, 1, 4),
            (Brightness, 0.5, 1.5), 
            (AddToHue, -255.0, 255.0),
            (Blur, 0, 15),
            (Noise, 1, 2),
            (Clouds,0.5, 1.0)
                    ]
def remove1box(bb_union, bb):
    
    if (bb_union.x2 == bb.x2 and bb_union.y2== bb.y2):
       x2 = bb.x1
       y2 = bb_union.y2
       x1 = bb_union.x1
       y1 = bb_union.y1
       x11 = bb.x1
       y11 = bb_union.y1
       x22 = bb_union.x2
       y22 = bb.y1
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = []
    elif (bb_union.x2 == bb.x2 and bb_union.y2 != bb.y2):
       x2 = bb.x1
       y2 = bb_union.y2
       x1 = bb_union.x1
       y1 = bb_union.y1
       y11 = bb.y2
       x11 = bb.x1
       x22 = bb_union.x2
       y22 = bb_union.y2
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = []

    elif (bb_union.x1==bb.x1 and bb_union.y1==bb.y1):
       x1 = bb.x2
       y1 = bb_union.y1
       x2 = bb_union.x2
       y2 = bb_union.y2
       x11 = bb_union.x1
       y11 = bb.y2
       x22 = bb.x2
       y22 = bb_union.y2
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = []
    elif(bb_union.x1 == bb.x1 and bb_union.y1 != bb.y1):
        x1 = bb.x2
        y1 = bb_union.y1
        x2 = bb_union.x2
        y2 = bb_union.y2
        x11 = bb_union.x1
        y11 = bb_union.y1
        x22 = bb.x2
        y22 = bb.y1   
        newbox1 = BoundingBox(x1, y1, x2, y2)
        newbox2 = BoundingBox(x11, y11, x22, y22)
        newbox3 = []
    elif(bb_union.x2 == bb.x2):
        x2 = bb.x1
        y2 = bb_union.y2
        x1 = bb_union.x1
        y1 = bb_union.y1
        newbox1 = BoundingBox(x1, y1, x2, y2)
        newbox2 = []
        newbox3 = []
    else:
        x1 = bb.x2 
        y1 = bb_union.y1
        x2 = bb_union.x2
        y2 = bb_union.y2  
        newbox1 = BoundingBox(x1, y1, x2, y2)
        newbox2 = []
        newbox3 = []
        
    return newbox1,newbox2,newbox3
def remove2box(bb_union,bb1,bb2):
    
    if(bb_union.x1 == bb1.x1 and bb_union.y1 == bb1.y1 and bb_union.x2 == bb2.x2 and bb_union.y2==bb2.y2):
       x1 = bb1.x2
       y1 = bb_union.y1
       x2 = bb2.x1
       y2 = bb_union.y2
       x11 = bb_union.x1
       y11 = bb1.y2
       x22 = bb1.x2
       y22 = bb_union.y2
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = []
       
    elif(bb_union.x1 == bb1.x1 and bb_union.y1 != bb1.y1 and bb_union.x2 == bb2.x2 and bb_union.y2==bb2.y2):
       x1 = bb1.x2
       y1 = bb_union.y1
       x2 = bb2.x1
       y2 = bb_union.y2
       x11 = bb_union.x1
       y11 = bb_union.y1
       x22 = bb1.x2
       y22 = bb1.y1
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = []
      
    elif(bb_union.x1 == bb1.x1 and bb_union.y1 == bb1.y1 and bb_union.x2 == bb2.x2 and bb_union.y2!=bb2.y2):
       x1 = bb1.x2
       y1 = bb_union.y1
       x2 = bb2.x1
       y2 = bb_union.y2
       x11 = bb2.x1
       y11 = bb2.y2
       x22 = bb_union.x2
       y22 = bb_union.y2
       x3 = bb2.x1
       y3 = bb_union.y1
       x4 = bb2.x2
       y4 = bb2.y1
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = BoundingBox(x3, y3, x4, y4)
       
    elif(bb_union.x1==bb2.x1 and bb_union.y1 == bb1.y1 and bb_union.x2==bb1.x2 and bb_union.y2==bb1.y2):
       x1 = bb2.x2
       y1 = bb_union.y1
       x2 = bb1.x1
       y2 = bb_union.y2
       x11 = bb_union.x1
       y11 = bb2.y2
       x22 = bb2.x2
       y22 = bb_union.y2
       x3 = bb1.x1
       y3 = bb_union.y1
       x4 = bb_union.x2
       y4 = bb1.y1
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = BoundingBox(x3, y3, x4, y4)
    elif(bb_union.x1==bb2.x1 and bb_union.y1 == bb2.y1 and bb_union.x2==bb1.x2 and bb_union.y2==bb2.y2):
       x1 = bb2.x2
       y1 = bb_union.y1
       x2 = bb1.x1
       y2 = bb_union.y2
       x11 = bb1.x1
       y11 = bb_union.y1
       x22 = bb_union.x2
       y22 = bb1.y1
       x3 = bb1.x1
       y3 = bb1.y2
       x4 = bb_union.x2
       y4 = bb_union.y2
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = BoundingBox(x3, y3, x4, y4)
    elif(bb_union.x1==bb2.x1 and bb_union.y1 == bb1.y1 and bb_union.x2==bb1.x2 and bb_union.y2==bb2.y2):
       x1 = bb2.x2
       y1 = bb_union.y1
       x2 = bb1.x1
       y2 = bb_union.y2
       x11 = bb_union.x1
       y11 = bb2.y2
       x22 = bb2.x2
       y22 = bb_union.y2
       x3 = bb_union.x1
       y3 = bb1.y1
       x4 = bb1.x2
       y4 = bb1.y1
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = BoundingBox(x11, y11, x22, y22)
       newbox3 = BoundingBox(x3, y3, x4, y4)
      
    else:
       x1 = bb1.x2
       y1 = bb_union.y1
       x2 = bb2.x1
       y2 = bb_union.y2
       newbox1 = BoundingBox(x1, y1, x2, y2)
       newbox2 = []
       newbox3 = []
       
    return newbox1, newbox2, newbox3

def conlist(bb_rem):
    bb_rem = [
                bb_rem.x1,
                bb_rem.y1,
                bb_rem.x2,
                bb_rem.y2
            ]
             
    return bb_rem
def imgresize(img,boxes):
    images_resized = ia.imresize_single_image(img, (480, 640))
    bbs = BoundingBoxesOnImage(
            [
                BoundingBox(*bb)
                for bb in boxes
                
            ],
            img.shape
           )
    #ia.imshow(bbs.draw_on_image(img))
    bbs_rescaled = bbs.on(images_resized)
    Bboxs = []     
   # indexval = 0
    for bb in bbs_rescaled:
        
        ##print("bb in rescaled",bbs_rescaled[indexval])
            Bbox = conlist(bb)
            
            Bboxs.append(Bbox)

    return images_resized,Bboxs

def conarray(bb_rem):
    bb_rem = np.array([
            [
                bb_rem.x1,
                bb_rem.y1,
                bb_rem.x2,
                bb_rem.y2
            ]
             ]).astype('int32')
    return bb_rem

def contextarea(bb_remb,bb_remb1,bb_remb2):
   if bb_remb:
      #ia.imshow(bb_remb.draw_on_image(image,size=2))
      bb_remb = conarray(bb_remb)
      
   if bb_remb1:
      #ia.imshow(bb_remb1.draw_on_image(image,size=2))
      bb_remb1 = conarray(bb_remb1)
   if bb_remb2:
      #ia.imshow(bb_remb2.draw_on_image(image,size=2))
      bb_remb2 = conarray(bb_remb2)

   if len(bb_remb)>0 and not len(bb_remb1)>0 and not len(bb_remb2)>0:
      bb_remb = bb_remb

   if len(bb_remb1)>0 and not len(bb_remb2)>0:
      bb_remb = np.vstack((bb_remb, bb_remb1))
   if len(bb_remb1)>0 and len(bb_remb2)>0:
      bb_remb = np.vstack((bb_remb, bb_remb1, bb_remb2))
   return bb_remb

def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    # if len(annos) and "segmentation" in annos[0]:
    #     segms = [obj["segmentation"] for obj in annos]
    #     if mask_format == "polygon":
    #         try:
    #             masks = PolygonMasks(segms)
    #         except ValueError as e:
    #             raise ValueError(
    #                 "Failed to use mask_format=='polygon' from the given annotations!"
    #             ) from e
    #     else:
    #         assert mask_format == "bitmask", mask_format
    #         masks = []
    #         for segm in segms:
    #             if isinstance(segm, list):
    #                 # polygon
    #                 masks.append(polygons_to_bitmask(segm, *image_size))
    #             elif isinstance(segm, dict):
    #                 # COCO RLE
    #                 masks.append(mask_util.decode(segm))
    #             elif isinstance(segm, np.ndarray):
    #                 assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
    #                     segm.ndim
    #                 )
    #                 # mask array
    #                 masks.append(segm)
    #             else:
    #                 raise ValueError(
    #                     "Cannot convert segmentation of type '{}' to BitMasks!"
    #                     "Supported types are: polygons as list[list[float] or ndarray],"
    #                     " COCO-style RLE as a dict, or a binary segmentation mask "
    #                     " in a 2D numpy array of shape HxW.".format(type(segm))
    #                 )
    #         # torch.from_numpy does not support array with negative stride.
    #         masks = BitMasks(
    #             torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
    #         )
    #     target.gt_masks = masks

    # if len(annos) and "keypoints" in annos[0]:
    #     kpts = [obj.get("keypoints", []) for obj in annos]
    #     target.gt_keypoints = Keypoints(kpts)

    # target.gt_masks= None
    # target.gt_keypoints=None

    return target
