# Context-Aware Augmentations for Data-Efficient Object Detection

This is the PyTorch implementation of Context-Aware Augmentations for Data-Efficient Object Detection (Currently under review).

## Dataset
[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [BDD](https://bdd-data.berkeley.edu/) and [Synthetic Fruits](https://github.com/roboflow/synthetic-fruit-dataset)  datasets were utilized for all our experiments. 

## Requirements
We tested our code with the following package versions

```bash
pytorch 1.12.0
cudatoolkit 11.3.0
Detectron2 0.6

```

## Experiments on Pascal VOC
# InterAug
Example command to train a faster-rcnn model with only 10% of training data using InterAug augmentation protocol. 
`python -u interaug_pascalvoc.py --model faster_rcnn --seed 0 --percent 0.1 --augmentation_type InterAug`

Similarly for retinanet please run

`python -u interaug_pascalvoc.py --model retinanet --seed 0 --percent 0.1 --augmentation_type InterAug`

# BBAug
Example commands to train  using BBAug Augmentation protocol

`python -u interaug_pascalvoc.py --model faster_rcnn --seed 0 --percent 0.1 --augmentation_type BBAug`

`python -u interaug_pascalvoc.py --model retinanet --seed 0 --percent 0.1 --augmentation_type BBAug`

# ImageAug

Sample commands to train using ImageAug augmentation strategy

`python -u imageaug_pascalvoc.py --model faster_rcnn --seed 0 --percent 0.1 --augmentation_type ImageAug`

`python -u imageaug_pascalvoc.py --model retinanet --seed 0 --percent 0.1 --augmentation_type ImageAug`

## Experiments on BDD

Sample commands to run on BDD using all three augmentation protocols. Please change the `--model` argument to desired architecture 

`python -u interaug_bdd.py --model faster_rcnn --seed 0 --percent 0.1`

`python -u bbaug_bdd.py --model faster_rcnn --seed 0 --percent 0.1`

`python -u imageaug_bdd.py --model faster_rcnn --seed 0 --percent 0.1`



