
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse
import copy
import os
import random
import warnings

import cv2
import detectron2
# Commented out IPython magic to ensure Python compatibility.
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
# import some common libraries
# import some common libraries
import numpy as np
import torch
import torchvision
# import some common detectron2 utilities
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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOEvaluator
# import some common detectron2 utilities
from detectron2.model_zoo import model_zoo
from detectron2.structures import (BitMasks, Boxes, BoxMode, Keypoints,
                                   PolygonMasks, RotatedBoxes, boxes)
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
# %matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from matplotlib.image import BboxImage
from PIL import Image

from utils.utils_aug_final import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")
import copy
import os
import random
import warnings

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
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import (register_coco_instances,
                                      register_pascal_voc,load_coco_json)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOEvaluator
# import some common detectron2 utilities
from detectron2.model_zoo import model_zoo
from detectron2.structures import (BitMasks, Boxes, BoxMode, Keypoints,
                                   PolygonMasks, RotatedBoxes, boxes)
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
# %matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from matplotlib.image import BboxImage
from PIL import Image


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
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset',type=str,default='BDD')
    parser.add_argument('--OUTPUT_DIR',type=str, default="/p/lustre2/thopalli/obj_detection/checkpoints/BDD/fixed_10_epochs/")
    parser.add_argument('--model',type=str, choices=['faster_rcnn','retinanet'],default='faster_rcnn')
    parser.add_argument('--seed',type=str,default='1')
    parser.add_argument('--percent',type=float,default=0.1)
   
    args = parser.parse_args()
    print(args)
    # register_coco_instances("trainset8_BDD", {}, "/home/kowshik/object_detection/DATA/bdd/annotations/BDD100ktrainjsn/split10percent_train.json",\
    #     "/home/kowshik/object_detection/DATA/bdd/bdd100k/images/100k/train")
    string = '_seed_'+str(args.seed)+'_per_'+str(args.percent)
    if args.dataset == 'BDD':
        
        train_json_file=os.path.join("/p/lustre2/thopalli/BDD/annotations/bdd_splits","train2017"+string+'.json')

        train_imgs_dir = "/p/lustre2/thopalli/BDD/bdd100k/images/100k/train"
        val_json_file= "/p/lustre2/thopalli/BDD/annotations/label/val.json"
        val_imgs_dir =  "/p/lustre2/thopalli/BDD/bdd100k/images/100k/val"

    else:

        
        
        train_json_file=os.path.join("/home/kowshik/object_detection/utils/synthetic_splits_train","train2017"+string+'.json')
        train_imgs_dir = "/home/kowshik/object_detection/DATA/synthetic_fruit/train"
        val_json_file= "/home/kowshik/object_detection/DATA/synthetic_fruit/annotations/validation_annotations.coco.json"
        val_imgs_dir= "/home/kowshik/object_detection/DATA/synthetic_fruit/valid"
    seed = int(args.seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    register_coco_instances("trainset", {}, train_json_file, train_imgs_dir)


    register_coco_instances("valset", {}, val_json_file,val_imgs_dir)

   # register_coco_instances("testset_BDD", {}, " ", "/home/kowshik/object_detection/DATA/bdd/bdd100k/images/100k/test")
    training_dict = load_coco_json(train_json_file, train_imgs_dir,
                dataset_name="trainset")
    training_metadata = MetadataCatalog.get("trainset")
    len_training_images = len(training_dict)
    number_of_epochs = 10
    
    cfg = get_cfg()
    model_file = "COCO-Detection/"+args.model+"_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
 
    cfg.DATASETS.TRAIN = ("trainset", )
    cfg.DATASETS.TEST = ("valset",)
    cfg.DATALOADER.NUM_WORKERS =4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.WARMUP_ITERS = 500
    max_iterations = int( len_training_images*number_of_epochs/cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = max_iterations   #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS =(int(max_iterations*0.4),int(max_iterations*0.7),int(max_iterations*0.9))
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.CHECKPOINT_PERIOD = int(len_training_images/cfg.SOLVER.IMS_PER_BATCH)
    if 'faster' in args.model:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE =512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(training_metadata.thing_classes) 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5     
    else:
        cfg.MODEL.RETINANET.NUM_CLASSES =len(training_metadata.thing_classes)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    print(cfg)
    num_gpu = 1
    bs = (num_gpu * 2)
    cfg.num_gpus=num_gpu
 # pick a good LR
    cfg.TEST.EVAL_PERIOD = int(len_training_images/cfg.SOLVER.IMS_PER_BATCH)
    cfg.OUTPUT_DIR=os.path.join(args.OUTPUT_DIR,args.model,'imageaug',str(args.seed), str(args.percent)+"percent/")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = AugTrainer(cfg)
    trainer.resume_or_load(resume= False)
    
    trainer.train()


    cfg.DATASETS.TEST = ("valset",)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("valset", cfg, False, output_dir=cfg.OUTPUT_DIR,use_fast_impl=False)
    DetectionCheckpointer(trainer.model).load(cfg.MODEL.WEIGHTS)
    val_loader = build_detection_test_loader(cfg, "valset")
    foo= inference_on_dataset(trainer.model, val_loader, evaluator)
    print(foo)