# AmongUsPlayerTracker
 Among Us player tracking with YOLOv5 and SIFT

This repository tracks player positions in the game Among Us, using SIFT feature mapping to localize your position on the map, and a YOLOv5 neural network model to detect players and their color. The YOLO detections are combined with SIFT map localization to plot player locations on the map.

This repo works in combination with the Ultralytics YOLOv5 repository. Clone it here: https://github.com/ultralytics/yolov5

### Tracking Example
- The green lines represent feature matches
- The blue rectangle represents where SIFT thinks the player's camera is located on the map
- The colored dots represet YOLOv5 detections plotted on the map
<a href="https://www.youtube.com/watch?v=ywalGN4IPnk"><img src="https://user-images.githubusercontent.com/47000850/94505551-54f7f780-01d9-11eb-992b-845050a898b8.png" alt="image" width="700" /></a>

### Running the Player Tracker
This is still a little rough right now - it's a two step process.
1. Run yolov5/detect.py to run YOLO on your video / image: 
`python detect.py --weights "yolov5\runs\exp___\weights\best.pt" --source "PATH/TO/YOUR/VIDEO" --img-size 1376 --device 0 --agnostic-nms --save-txt`

2. Run `AmongUsPlayerTracker/sift_template_matcher.py` to do the player tracking and plotting. Use the appropriate map image from the AmongUsPlayerTracker/map_images folder. 
`python sift_template_matcher.py --map-img "map_images/the_skeld.png" --video "PATH/TO/YOUR/VIDEO.mp4" --yolo-output "yolov5\inference\output"`

---------------------------------------------------------------------------------------------------------------
## Results So Far

### SIFT Algorithm Performance
SIFT matching is working pretty well when combined with some constraints on results. As you can see in the video above, it loses tracking when the player brings up the menus or does tasks, but otherwise tracks the player's camera perfectly.

### YOLOv5s Algorithm Performance
This algorithm works extremely well in normal circumstances but has trouble in the following scenarios:
- The lights are sabotaged: When the lights are turned off, the player's color RGB values change, often causing the model to detect the wrong color.
- The reactor/oxygen is sabotaged: The flashing red light again often causes the model to detect the wrong color
- Many players are stacked on top of each other. This really hard for a human to see anyway....

The lights/reactor/oxygen issues might be solved with a data augmentation model that could accurately represent the color changing effects, or simply more data. This assumes of course that the YOLOv5s model has the capability to handle these cases. The stacking issue seems pretty hard to solve, but in practice might not be necessary, because you could track the players before/after they entered the stack?

---------------------------------------------------------------------------------------------------------------
## YOLOv5 Training

### Data Prep
1. Capture training data - to do this, I grabbed ~160 images from various Among Us games recorded on Youtube and stored them into the `/data` folder
2. Label training data - I used the tool Labelme (https://github.com/wkentaro/labelme) to generate bbox data stored in json files
3. Run yolo_data_prep.py - This will convert the images in the `/data` folder to darknet format described here https://github.com/ultralytics/yolov5/issues/12. You can use --data "path/to/data/folder" --outdir "darknet/data/output/folder" if you want different input/output folders
4. Modify the train and val paths in `/data_darknet_format/amongus.yaml` to the appropriate full paths to your data

### Neural Network Training
I prefer to use the default YOLOv5s model as it runs at ~60FPS on image size 1376 and seems to be able to perform well. I changed the data augmentation settings, as some of the augmentations for MS COCO reduce performance, particularly the HSV augmentation. The augmentation settings I found to work well are in `/data_darknet_format/hyp.yaml`

Use YOLOv5/train.py to train a YOLOv5s model from scratch. For my 150 image dataset, I used the following command to train at max image size with batch size 8 for 100 epochs: `python train.py --weights "" --cfg ./models/yolov5s.yaml --data "path\to\your\data_darknet_format\amongus.yaml" --img-size 1376 --batch-size 8 --epochs 100 --device 0 --rect --hyp "path\to\your\hyp.yaml"`

Training will save your trained models to the `yolov5\runs\exp_____\weights` folder.

### Neural Network Inference
Use yolov5/detect.py for inference:
`python detect.py --weights "yolov5\runs\exp___\weights\best.pt" --source "PATH/TO/YOUR/VIDEO" --img-size 1376 --device 0 --agnostic-nms --save-txt`
