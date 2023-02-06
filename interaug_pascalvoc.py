
import copy
import os
import random
import warnings
import pickle
import cv2
import detectron2
# Commented out IPython magic to ensure Python compatibility.
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
# import some common libraries
import numpy as np
import torch
import torchvision
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import (register_coco_instances,
                                      register_pascal_voc)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.model_zoo import model_zoo
from detectron2.structures import (BitMasks, Boxes, BoxMode, Keypoints,
                                   PolygonMasks, RotatedBoxes, boxes)
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
# %matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from matplotlib.image import BboxImage
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, PascalVOCDetectionEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from utils.utils_aug_final import *
import argparse
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
class RandAugment:
  
      def __call__(self, image, boxes):
        ops_all = random.choices(ALL_TRANSFORMS, k=1)
        #print(ops_all)
               
        for ops in ops_all:
          #print(type(image))
          op= ops[0]
          minval = ops[1]
          maxval=ops[2]
          
          level =np.random.uniform(minval,maxval,1)[0]

          #print(op,minval,maxval,level)
          img = op(image, boxes,level)
                  
        return img

ALL_boxchoices = ['boxunion', 'remove1box1', 'remove2box2']
#ALL_boxchoices = ['RandAugment1', 'remove1box1', 'remove2box2']

class InterAug:
   
    def __call__(self, img, boxs):
        #return_dict={}        
        augch = np.random.choice(ALL_boxchoices)
        #print(augch)
        #return_dict['augch']=augch
        bbs = BoundingBoxesOnImage(
            [
                BoundingBox(*bb)
                for bb in boxs
                
            ],
            img.shape
        )
        if augch == 'RandAugment1':
           aug = RandAugment()
           aug_image = aug(img, boxs)

        elif augch == 'boxunion': 
           if len(bbs)>1:
              #print("length")
              #print(len(bbs))
              bb1= np.random.choice(bbs.bounding_boxes)
              #ia.imshow(bb1.draw_on_image(img,size=2))
              bb2= np.random.choice(bbs.bounding_boxes)
              #ia.imshow(bb2.draw_on_image(img,size=2))
              if bb1 == bb2:
                 bb2= np.random.choice(bbs.bounding_boxes)
              bb_union = bb1.union(bb2)
              #ia.imshow(bb_union.draw_on_image(img,size=2))
              bb_union = conarray(bb_union)
              aug = RandAugment()
              #print("aug=",aug)
              aug_image = aug(img,bb_union)
           else:
              #print("box length is 1")
              aug = RandAugment()
              #print(aug)
              aug_image = aug(img,boxs) 
         
        elif augch == 'remove1box1':
           if len(bbs)>1:
              #print("length")
              #print(len(bbs))
              bb1= np.random.choice(bbs.bounding_boxes)
              #ia.imshow(bb1.draw_on_image(img,size=2))
              bb2= np.random.choice(bbs.bounding_boxes)
              #ia.imshow(bb2.draw_on_image(img,size=2))
              if bb1 == bb2:
                 bb2= np.random.choice(bbs.bounding_boxes)
              bb_union = bb1.union(bb2)
              #ia.imshow(bb_union.draw_on_image(img,size=2))
              boxchoice = ['bb_remb1', 'bb_remb2']
              op = np.random.choice(boxchoice)
              #print(op)
              if op == 'bb_remb1':
                 bb_remb,bb_remb1,bb_remb2= remove1box(bb_union,bb1)
           
              else:
                 bb_remb,bb_remb1,bb_remb2 = remove1box(bb_union,bb2)
           
              bb_remb = contextarea(bb_remb, bb_remb1, bb_remb2)
              #print(bb_remb)
              aug = RandAugment()
              #print("aug=",aug)
              aug_image = aug(img,bb_remb)
                            
           else:
              #print("box length is 1")
              aug = RandAugment()
              #print(aug)
              aug_image = aug(img,boxs)                   
        else:
           if len(bbs)>1:
              bb1= np.random.choice(bbs.bounding_boxes)
              #ia.imshow(bb1.draw_on_image(img,size=2))
              bb2= np.random.choice(bbs.bounding_boxes)
              #ia.imshow(bb2.draw_on_image(img,size=2))
              if bb1 == bb2:
                 bb2= np.random.choice(bbs.bounding_boxes)
              bb_union = bb1.union(bb2)
             # ia.imshow(bb_union.draw_on_image(img,size=2))
              bb_remb,bb_remb1,bb_remb2 = remove2box(bb_union, bb1,bb2)
           
              bb_remb = contextarea(bb_remb, bb_remb1, bb_remb2)
                #print(bb_remb)
              aug = RandAugment()
              aug_image = aug(img,bb_remb)  
           else:
              #print("box length is 1 when remove2box")
              aug = RandAugment()
              #print(aug)
              aug_image = aug(img,boxs)   
             
        return aug_image

def custom_mapper(dataset_dicts):
    dataset_dict = copy.deepcopy(dataset_dicts)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    utils.check_image_size(dataset_dict, image)
    prev_anno = dataset_dict["annotations"]
    boxes = np.array([obj["bbox"] for obj in prev_anno], dtype=np.float32)   
    resized_img, resized_bboxes = imgresize(image,boxes)   
    aug = InterAug()
    image = aug(resized_img, resized_bboxes)
    #image = aug_dict['aug_image']  
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    #image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    category_id2 = np.arange(len(dataset_dict["annotations"]))
    annos = []
    for i, j in enumerate(category_id2):
        d = prev_anno[j]
        
        d["bbox"] = resized_bboxes[i]
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
def custom_mapper_bbaug(dataset_dicts):
    dataset_dict = copy.deepcopy(dataset_dicts)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    utils.check_image_size(dataset_dict, image)
    prev_anno = dataset_dict["annotations"]
    boxes = np.array([obj["bbox"] for obj in prev_anno], dtype=np.float32)
    aug = RandAugment()
    image = aug(image, boxes)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    #image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    category_id2 = np.arange(len(dataset_dict["annotations"]))
    annos = []
    for i, j in enumerate(category_id2):
        d = prev_anno[j]
        d["bbox"] = boxes[i]
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

class BBAugTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper_bbaug)


if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='Process some integers.')
   parser.add_argument('--augmentation_type', type=str,default = "InterAug", \
     choices=["InterAug","BBAug","NoAug","AugMix"])
   parser.add_argument('--train_dataset',default=['trainval2012','trainval2007'],nargs='+')
   parser.add_argument('--test_dataset',type = str, default="PASCAL2007")
   parser.add_argument('--OUTPUT_DIR',type=str, default="/p/lustre2/thopalli/obj_detection/checkpoints/pascalVOC_split_class")
   parser.add_argument('--seed',type=str,default='1')
   parser.add_argument('--percent',type=float,default=0.4)
   parser.add_argument('--model',type=str, choices=['faster_rcnn','retinanet'],default='retinannet')

   args = parser.parse_args()
   print(args)
   for seed in [int(args.seed)]:
      print(seed)
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
      if args.augmentation_type == 'InterAug':
         trainer = AugTrainer(cfg)
      elif args.augmentation_type == 'BBAug':
         trainer = BBAugTrainer(cfg)
      else:
         pass
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
      print('saving json')
      with open(os.path.join(cfg.OUTPUT_DIR,"pascal_results_in_coco.json"),"w") as f:
         f.write(json.dumps(results['bbox']['coco_preds']))
         f.flush()
      
