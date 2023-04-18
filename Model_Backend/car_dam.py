# %% [markdown]
# The left image shows the damages and right image shows the parts, both of these can be plotted using this notebook [https://www.kaggle.com/lplenka/coco-data-visualization](https://www.kaggle.com/lplenka/coco-data-visualization)

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'>Source Dataset</center></h3>

# %% [markdown]
# **Since we will train two models, first for only damages and second for only parts, you can find annotation for both in the dataset I have published here. [Coco Car Damage Dataset](https://www.kaggle.com/lplenka/coco-car-damage-detection-dataset)**

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'>Let's begin!</center></h3>

# %% [markdown]
# ##### Since I have already shown the installation steps, here I will directly start with all installations

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'>Installation</center></h3>

# %%
# # Install Pycocotools
# !pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# # Install detectron 2
# !python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'>Import Libraries</center></h3>

# %%
# %matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
pylab.rcParams['figure.figsize'] = (8.0, 10.0)# Import Libraries

# For visualization
import os
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image

# Scipy for calculating distance
from scipy.spatial import distance

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'>Set constant variables</center></h3>

# %%
# I am visualizing some images in the 'val/' directory

dataDir='./image/val'
dataType='COCO_val_annos'
mul_dataType='COCO_mul_val_annos'
annFile='{}/{}.json'.format(dataDir,dataType)
mul_annFile='{}/{}.json'.format(dataDir,mul_dataType)
img_dir = "./image/img"

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white; border:0' role="tab" aria-controls="home"><center style='padding-top: 15px'> Initialize the COCO API</center></h3>

# %%
# initialize coco api for instance annotations
coco=COCO(annFile)
mul_coco=COCO(mul_annFile)

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px' > Import Libraries required for training</center></h3>

# %%
# import torch, torchvision
# print(torch.__version__, torch.cuda.is_available())

# %%
# assert torch.__version__.startswith("1.7")

# %%
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Set base params
plt.rcParams["figure.figsize"] = [16,9]

# %%
# To find out inconsistent CUDA versions, if there is no "failed" word in this output then things are fine.
# !python -m detectron2.utils.collect_env

# %% [markdown]
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'>  Register Car Damage Dataset </center></h3>

# %% [markdown]
# #### Register Train Dataset, so that we can use its Metadata

# %%

dataset_dir = "./image/"
img_dir = "img/"
train_dir = "train/"
val_dir = "val/"

# %%
from detectron2.data.datasets import register_coco_instances
register_coco_instances("car_dataset_val", {}, os.path.join(dataset_dir,val_dir,"COCO_val_annos.json"), os.path.join(dataset_dir,img_dir))
register_coco_instances("car_mul_dataset_val", {}, os.path.join(dataset_dir,val_dir,"COCO_mul_val_annos.json"), os.path.join(dataset_dir,img_dir))

# %% [markdown]
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white; ' role="tab" aria-controls="home"><center style='padding-top: 15px'> Load trained model </center></h2>

# %% [markdown]
# #### I will load two pretained models:
# 
# * Damage Segmentation model weights -  This can be easily created using this notebook [
# Detectron2 Car Damage Detection](https://www.kaggle.com/lplenka/detectron2-car-damage-detection). The model is stored in default output directory.
# 
# * Parts Segmentation Model weights - This can be also created just changing the dataset from damage annotions to parts annotation in [cell 22](https://www.kaggle.com/lplenka/detectron2-car-damage-detection?scriptVersionId=52171508&cellId=37)
# 

# %% [markdown]
# 
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white; ' role="tab" aria-controls="home"><center style='padding-top: 15px'> Damage Detection Model </center></h2>

# %%
#get configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (damage) + 1
cfg.MODEL.RETINANET.NUM_CLASSES = 2 # only has one class (damage) + 1
cfg.MODEL.WEIGHTS = os.path.join("./damage_segmentation_model.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
cfg['MODEL']['DEVICE']='cpu'#or cpu
damage_predictor = DefaultPredictor(cfg)

# %% [markdown]
# 
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white; ' role="tab" aria-controls="home"><center style='padding-top: 15px'> Parts Segmentation Model </center></h2>

# %%
cfg_mul = get_cfg()
cfg_mul.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_mul.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has five classes (headlamp,hood,rear_bumper,front_bumper_door) + 1
cfg_mul.MODEL.RETINANET.NUM_CLASSES = 6 # only has five classes (headlamp,hood,rear_bumper,front_bumper_door) + 1
cfg_mul.MODEL.WEIGHTS = os.path.join("./part_segmentation_model.pth")
cfg_mul.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
cfg_mul['MODEL']['DEVICE']='cpu' #or cpu
part_predictor = DefaultPredictor(cfg_mul)

# %% [markdown]
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'> Model Inference </center></h2>

# %%
def detect_damage_part(damage_dict, parts_dict):
  """
  Returns the most plausible damaged part for the list of damages by checking the distance 
  between centers centers of damage_polygons and parts_polygons

  Parameters
  -------------
   damage_dict: dict
                Dictionary that maps damages to damage polygon centers.
   parts_dict: dict
                Dictionary that maps part labels to parts polygon centers.
  Return
  ----------
  part_name: str
            The most plausible damaged part name.
  """
  try:
    max_distance = 10e9
    assert len(damage_dict)>0, "AssertError: damage_dict should have atleast one damage"
    assert len(parts_dict)>0, "AssertError: parts_dict should have atleast one part"
    max_distance_dict = dict(zip(damage_dict.keys(),[max_distance]*len(damage_dict)))
    part_name = dict(zip(damage_dict.keys(),['']*len(damage_dict)))

    for y in parts_dict.keys():
        for x in damage_dict.keys():
          dis = distance.euclidean(damage_dict[x], parts_dict[y])
          if dis < max_distance_dict[x]:
            part_name[x] = y.rsplit('_',1)[0]

    return list(set(part_name.values()))
  except Exception as e:
    print(e)

# # %%

# damage_class_map= {0:'damage'}
# parts_class_map={0:'headlamp',1:'rear_bumper', 2:'door', 3:'hood', 4: 'front_bumper'}

# # %%
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize =(16,12))
# im = io.imread("./yolo/images (83).jpeg")

# #damage inference
# damage_outputs = damage_predictor(im)
# damage_v = Visualizer(im[:, :, ::-1],
#                    metadata=MetadataCatalog.get("car_dataset_val"), 
#                    scale=0.5, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
# )
# damage_out = damage_v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))

# #part inference
# parts_outputs = part_predictor(im)
# parts_v = Visualizer(im[:, :, ::-1],
#                    metadata=MetadataCatalog.get("car_mul_dataset_val"), 
#                    scale=0.5, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
# )
# parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))

# #plot
# ax1.imshow(damage_out.get_image()[:, :, ::-1],)
# ax2.imshow(parts_out.get_image()[:, :, ::-1])
# ax1.figure.savefig("output.jpg")

# # %% [markdown]
# # <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'> Create damage polygons </center></h2>

# # %% [markdown]
# # For now allowing multiple polygons of same class label

# # %%
# damage_prediction_classes = [ damage_class_map[el] + "_" + str(indx) for indx,el in enumerate(damage_outputs["instances"].pred_classes.tolist())]
# damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
# damage_dict = dict(zip(damage_prediction_classes,damage_polygon_centers))

# # %% [markdown]
# # <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'> Create parts polygons </center></h2>

# # %% [markdown]
# # For now allowing multiple polygons of same class label

# # %%

# parts_prediction_classes = [ parts_class_map[el] + "_" + str(indx) for indx,el in enumerate(parts_outputs["instances"].pred_classes.tolist())]
# parts_polygon_centers =  parts_outputs["instances"].pred_boxes.get_centers().tolist()



# #Remove centers which lie in beyond 800 units
# parts_polygon_centers_filtered = list(filter(lambda x: x[0] < 800 and x[1] < 800, parts_polygon_centers))
# parts_dict = dict(zip(parts_prediction_classes,parts_polygon_centers_filtered))

# # %% [markdown]
# # <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; color:white;' role="tab" aria-controls="home"><center style='padding-top: 15px'> Damaged Parts </center></h2>

# # %%
# print("Damaged Parts: ",detect_damage_part(damage_dict,parts_dict))

# %%
def predict_cond(img_pth):
    
    damage_class_map= {0:'damage'}
    parts_class_map={0:'headlamp',1:'rear_bumper', 2:'door', 3:'hood', 4: 'front_bumper'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize =(16,12))
    im = io.imread(img_pth)

    #damage inference
    damage_outputs = damage_predictor(im)
    damage_v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("car_dataset_val"), 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    damage_out = damage_v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))

    #part inference
    parts_outputs = part_predictor(im)
    parts_v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("car_mul_dataset_val"), 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))

    #plot
    ax1.imshow(damage_out.get_image()[:, :, ::-1],)
    ax2.imshow(parts_out.get_image()[:, :, ::-1])
    ax1.figure.savefig("output.jpg")
    damage_prediction_classes = [ damage_class_map[el] + "_" + str(indx) for indx,el in enumerate(damage_outputs["instances"].pred_classes.tolist())]
    damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
    damage_dict = dict(zip(damage_prediction_classes,damage_polygon_centers))
    parts_prediction_classes = [ parts_class_map[el] + "_" + str(indx) for indx,el in enumerate(parts_outputs["instances"].pred_classes.tolist())]
    parts_polygon_centers =  parts_outputs["instances"].pred_boxes.get_centers().tolist()



#Remove centers which lie in beyond 800 units
    parts_polygon_centers_filtered = list(filter(lambda x: x[0] < 800 and x[1] < 800, parts_polygon_centers))
    parts_dict = dict(zip(parts_prediction_classes,parts_polygon_centers_filtered))
    return("Damaged Parts: ",detect_damage_part(damage_dict,parts_dict))
    


