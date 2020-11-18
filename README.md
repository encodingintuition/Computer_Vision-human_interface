# Computer Vision - Hand Gesture Recognition 

## Executive Summary:

#### Opportunity Statement:

To create a more human interface to interact with a computer system.

- Stage 1 – to have a computer system interact with pre-defined gestures 
- Stage 2 – to have a model intuit human needs through our gestures ( as the family dog uses observation to reactively communicate) 


#### Project Goal: 

- To train a convolutional neural network to identify particular hand gestures.
- To have these hand gestures trigger actions within another python application


#### Hand guesture being reconized by model
<img src="./code/images/magic.jpg"
     alt="magic descovery"
     style="float: left; margin-right: 10px;" />

#### Therefore montage of shapes created triggered actions 

<img src="./code/images/turtle_shapes.jpg"
     alt="turtle images"
     style="float: left; margin-right: 10px;" />

#### Data:

- opensource online archive 
- ~15 partisipants via phone photos of their hands
- webcam screen shoot 

### Metrics 


| class          | Average Precision    | TP    |  FP |
| :------------- | :----------: | :----------: |-----------: |
|  foward | 90.00  | 9   | 2|
|  back | 100  | 10  | 0|
|  left| 83.33  | 5   | 0|
|  right| 91.82  | 10   | 2|
|  input| 90  | 5   | 1|
|  plus| 81.48  | 8   | 1|
|  three| 100  | 5   | 2|
|  seven | 100  | 7   | 1|
|five | 100| 2 |0 \||

| Average mean precision (mAP)         |recall    | precision   | F1-score |
| :------------- | :----------: | :----------: |-----------: |
|  92.96 | 0.91  | 0.87   | 0.89|

#### Findings

A YoloV4 model is successful in being able to identify particular hand gestures after being trained on custom data.

The detection is able to take place in real time and because of the architecture 
of the YoloV4 model there is not a significant computational cost put on hardware. 


### Next steps
- implementing this project for education and introduction into computer vision to further democratize CV
- creating better tools to manage data workflow
- Look into other models such as Mediapipe (trained on 30K hands)
- Look into passively collecting data, ‘clustering’ for body movements.  
### Next next steps
- To examine the implications and uses of computer intuiting full body gestures
- Gesture types
 - Public / shared dictonary of gestures, ex: hand wave, a smile, head node.
 - Private / personal gestures/  This vocabulary of gestures would give deep insight into who you are and how you act.

--- 

---

# Community notes
- Below are notes from the yolo community that are most helpful when creating a model. 
- One of the greatest strengths of the yolo model is the Opensource community that supports it.

#### About YoloV4 tiny 

https://github.com/AlexeyAB/darknet/issues/6067

---

### People to reconize in the comunity

Yashas Samaga B L

Worked to create YOLOv4 inference using OpenCV DNN 

https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49

---

### Hyperparameters

https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section

---

## Best Practices for training 

#### Below are notes from   https://github.com/AlexeyAB/darknet#How-to-improve-object-detection

- Check that each object that you want to detect is mandatory labeled in your dataset - no one object in your data set should not be without label. In the most training issues - there are wrong labels in your dataset (got labels by using some conversion script, marked with a third-party tool, ...). Always check your dataset by using: https://github.com/AlexeyAB/Yolo_mark

- for each object which you want to detect - there must be at least 1 similar object in the Training dataset with about the same: shape, side of object, relative size, angle of rotation, tilt, illumination. So desirable that your training dataset include images with objects at diffrent: scales, rotations, lightings, from different sides, on different backgrounds - you should preferably have 2000 different images for each class or more, and you should train 2000*classes iterations or more

- desirable that your training dataset include images with non-labeled objects that you do not want to detect - negative samples without bounded box (empty .txt files) - use as many images of negative samples as there are images with objects

- If you train the model to distinguish Left and Right objects as separate classes (left/right hand, left/right-turn on road signs, ...) then for disabling flip data augmentation - add flip=0 here: https://github.com/AlexeyAB/darknet/blob/3d2d0a7c98dbc8923d9ff705b81ff4f7940ea6ff/cfg/yolov3.cfg#L17

- General rule - your training dataset should include such a set of relative sizes of objects that you want to detect:

    train_network_width * train_obj_width / train_image_width ~= detection_network_width * detection_obj_width / detection_image_width
    train_network_height * train_obj_height / train_image_height ~= detection_network_height * detection_obj_height / detection_image_height

I.e. for each object from Test dataset there must be at least 1 object in the Training dataset with the same class_id and about the same relative size:

object width in percent from Training dataset ~= object width in percent from Test dataset

That is, if only objects that occupied 80-90% of the image were present in the training set, then the trained network will not be able to detect objects that occupy 1-10% of the image.

- each: model of object, side, illimination, scale, each 30 grad of the turn and inclination angles - these are different objects from an internal perspective of the neural network. So the more different objects you want to detect, the more complex network model should be used.

- to make the detected bounded boxes more accurate, you can add 3 parameters ignore_thresh = .9 iou_normalizer=0.5 iou_loss=giou to each [yolo] layer and train, it will increase mAP@0.9, but decrease mAP@0.5.

After training - for detection:

- Increase network-resolution by set in your .cfg-file (height=608 and width=608) or (height=832 and width=832) or (any value multiple of 32) - this increases the precision and makes it possible to detect small objects: link


#### About anchers

- https://github.com/pjreddie/darknet/issues/911

- Only if you are an expert in neural detection networks - recalculate anchors for your dataset for width and height from cfg-file: darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416 then set the same 9 anchors in each of 3 [yolo]-layers in your cfg-file. But you should change indexes of anchors masks= for each [yolo]-layer, so for YOLOv4 the 1st-[yolo]-layer has anchors smaller than 30x30, 2nd smaller than 60x60, 3rd remaining, and vice versa for YOLOv3. Also you should change the filters=(classes + 5)*<number of mask> before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.

#### Mosaic Augmentation Paper? #8

https://github.com/WongKinYiu/CrossStagePartialNetworks/issues/8
