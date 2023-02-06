import argparse
import copy
import os
import random
import warnings
import pickle
import cv2
import detectron2
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import (register_coco_instances,
                                      register_pascal_voc)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import (COCOEvaluator, PascalVOCDetectionEvaluator,
                                   inference_on_dataset)
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.model_zoo import model_zoo
from detectron2.structures import (BitMasks, Boxes, BoxMode, Keypoints,
                                   PolygonMasks, RotatedBoxes, boxes)
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from matplotlib.image import BboxImage
from PIL import Image

from utils.utils_aug_final import *


def Solarize(img, level):  # [0, 256]  
    solarize = iaa.Solarize(level, threshold=(32, 128))
    img_aug = solarize(image=img)
    return img_aug

def auto_contrast(img,level):
    auto_cont = iaa.pillike.Autocontrast(level)
    img_aug = auto_cont(image=img)
    return img_aug

def emboss(img, level):
    emboss = iaa.Emboss(alpha=(0, 1.0), strength=level)
    img_aug = emboss(image=img)
    return img_aug


def Enhancecolor(img,level):
    enhancecolor = iaa.pillike.EnhanceColor(level)
    img_aug = enhancecolor(image=img)
    return img_aug

def EnhanceSharpness(img, level):
    enhanceSharpness = iaa.pillike.EnhanceSharpness(level)
    img_aug = enhanceSharpness(image=img)
    return img_aug

def Posterize(img,level): 
    posterize = iaa.pillike.Posterize(int(level))
    img_aug = posterize(image=img)
    return img_aug

def Enhancecontrast(img, level):
    enhancecontrast = iaa.pillike.EnhanceContrast(level)
    img_aug = enhancecontrast(image=img)
    return img_aug

def Brightness(img, level):
    brightness = iaa.pillike.EnhanceBrightness(level)
    img_aug = brightness(image=img)
    return img_aug
 
def AddToHue(img, level):
    addtohue = iaa.AddToHue((-255,255))
    img_aug = addtohue(image=img)
    return img_aug

def Noise(img, level):
    noise = iaa.OneOf([iaa.imgcorruptlike.GaussianNoise(severity=2),iaa.imgcorruptlike.ImpulseNoise(severity=2),iaa.imgcorruptlike.ShotNoise(severity=2)])
    img_aug = noise(image=img)
    return img_aug

def Clouds(img, level):
    cloudaug = iaa.BlendAlphaMask(iaa.InvertMaskGen(level, iaa.VerticalLinearGradientMaskGen()),iaa.Clouds())    
    img_aug = cloudaug(image=img)
    return img_aug

def fliplr(img, level):
    flip = iaa.Fliplr(level)
    img_aug = flip(image=img)
    return img_aug

def centercrop(img, level):
    crop = iaa.CenterCropToAspectRatio(level)
    img_aug = crop(image=img)
    return img_aug

def Blur(img, level):
    blur = iaa.OneOf([iaa.GaussianBlur(sigma =(0.0, 3.0)),iaa.MedianBlur(k=(3, 7)),iaa.MotionBlur(k=15)])
    img_aug = blur(image=img)
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
            (Posterize, 1.0, 4.0),
            (Brightness, 0.5, 1.5), 
            (AddToHue, -255.0, 255.0),
            (Blur, 0, 15),
            (Noise, 1, 2),
            (Clouds,0.5, 1.0)
                    ]

class RandAugment:
  
      def __call__(self, image):
        ops_all = random.choices(ALL_TRANSFORMS, k=1)
        #print(ops_all)
               
        for ops in ops_all:
          ##print(type(image))
          op= ops[0]
          minval = ops[1]
          maxval=ops[2]
          
          level =np.random.uniform(minval,maxval,1)[0]

          #print(op,minval,maxval,level)
          img = op(image, level)
                  
        return img

ALL_boxchoices = ['boxunion', 'remove1box1', 'remove2box2']

def custom_mapper(dataset_dicts):
    dataset_dict = copy.deepcopy(dataset_dicts)  # it will be modified by code below # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    utils.check_image_size(dataset_dict, image)
    
    pre_annos = dataset_dict.get("annotations", None)
    if pre_annos:
      boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in pre_annos
            ]
            
    aug = RandAugment()
    image = aug(image)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    category_id2 = np.arange(len(dataset_dict["annotations"]))
    annos = []
    for i, j in enumerate(category_id2):
        d = pre_annos[j]
        d["bbox"] = boxes[i]
        d["bbox_mode"] = BoxMode.XYXY_ABS
        annos.append(d)
    dataset_dict.pop("annotations", None)  # Remove unnecessary field. 
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    #return dataset_dict
    return {
       # create the format that the model expects
       "image": dataset_dict["image"],
       "instances": dataset_dict["instances"],
       "width": image.shape[0],
       "height":image.shape[1]
   }


class AugTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--augmentation_type',type=str,default = "ImageAug")
    parser.add_argument('--train_dataset',default=['trainval2012','trainval2007'],nargs='+')
    parser.add_argument('--test_dataset',type = str, default="PASCAL2007")
    parser.add_argument('--OUTPUT_DIR',type=str, default="/p/lustre2/thopalli/obj_detection/checkpoints/pascalVOC_split_class")
    parser.add_argument('--seed',type=str,default='2')
    parser.add_argument('--percent',type=float,default=1.0)
    parser.add_argument('--model',type=str, choices=['faster_rcnn','retinanet'],default='retinanet')
 
    args = parser.parse_args()
    print(args)
    for seed in [int(args.seed)]:
        #seed = int(args.seed)
        seed_arg =seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
        CLASS_NAMES = (
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    
        per= args.percent
        string = '_per_'+str(per)+'_seed_'+str(seed_arg)#+'_notsplit_class'
        
        root_2012 = "/p/lustre2/thopalli/detectron2_datasets/VOC/DATA/VOCdevkit/VOC2012"
        root_2007 = "/p/lustre2/thopalli/detectron2_datasets/VOC/DATA/VOCdevkit/VOC2007"
    
        if per ==1.0:
            register_pascal_voc("trainval_2012", root_2012, "trainval", 2012, class_names=CLASS_NAMES)
            register_pascal_voc("trainval_2007", root_2007, "trainval", 2007, class_names=CLASS_NAMES)
        
        else:
        
            # register_pascal_voc("trainval_2012", root_2012, "trainval"+string+"_notsplit_class", 2012, class_names=CLASS_NAMES)
            # register_pascal_voc("trainval_2007", root_2007, "trainval"+string+"_notsplit_class", 2007, class_names=CLASS_NAMES)

            register_pascal_voc("trainval_2012", root_2012, "trainval"+string, 2012, class_names=CLASS_NAMES)
            register_pascal_voc("trainval_2007", root_2007, "trainval"+string, 2007, class_names=CLASS_NAMES)
    
        register_pascal_voc("testset_2007", root_2007, "test", 2007, class_names=CLASS_NAMES)
        
        cfg = get_cfg()
        model_file = "COCO-Detection/"+args.model+"_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(model_file))
        #cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
        cfg.DATASETS.TRAIN = ("trainval_2012", "trainval_2007")
        cfg.DATASETS.TEST = ("testset_2007", )
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
        cfg.SOLVER.IMS_PER_BATCH = 32
        #cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.WARMUP_ITERS = 500
        cfg.SOLVER.MAX_ITER = 10000   #adjust up if val mAP is still rising, adjust down if overfit
        #cfg.SOLVER.STEPS = (100, 1200)
        cfg.SOLVER.STEPS = ( 5000,7500)#1200)
        
        cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
        cfg.MODEL.RETINANET.NUM_CLASSES =len(CLASS_NAMES)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
        num_gpu = 1
        bs = (num_gpu * 2)
        cfg.num_gpus=num_gpu
        cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
        #cfg.TEST.EVAL_PERIOD = 500
        cfg.OUTPUT_DIR=os.path.join(args.OUTPUT_DIR,args.model,args.augmentation_type,str(seed), str(args.percent)+"percent/")
        #cfg.OUTPUT_DIR=os.path.join(args.OUTPUT_DIR,string)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        trainer = AugTrainer(cfg)
    
        trainer.resume_or_load(resume= False)
        trainer.train()
        #10000 poscal evaluator
        cfg.DATASETS.TEST = ("testset_2007",)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
        predictor = DefaultPredictor(cfg)
        evaluator = PascalVOCDetectionEvaluator("testset_2007")
        val_loader = build_detection_test_loader(cfg, "testset_2007")
        DetectionCheckpointer(trainer.model).load(cfg.MODEL.WEIGHTS)
        results=inference_on_dataset(trainer.model, val_loader, evaluator)
        print('AP50',results['bbox']['AP50'])
        print('AP50_PERCATEGORY',results['bbox']['AP50_PERCATEGORY'])
        with open (os.path.join(cfg.OUTPUT_DIR,"results_preds_Ap_50_percategory.pkl"),"wb") as f:
             pickle.dump(results,f)
        import json
        with open(os.path.join(cfg.OUTPUT_DIR,"pascal_results_in_coco.json"),"w") as f:
            f.write(json.dumps(results['bbox']['coco_preds']))
            f.flush()