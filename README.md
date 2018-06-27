# data_clean2voc
The repository consists of some scripts that can clean the annotations to xml files

---
#### For DETRAC dataset
```
You can download it by using the following URL.
https://detrac-db.rit.albany.edu/download
```

---
#### Training YOLO on VOC
##### Get The Pascal VOC Data
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
##### Generate Labels for VOC
```
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Modify Cfg for Pascal Data
Change the cfg/voc.data config file
```
train  = train.txt
valid  = 2007_test.txt
names = data/voc.names
backup = backup
```
##### Download Pretrained Convolutional Weights
Download weights from the convolutional layers
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```
or run the following command:
```
python partial.py cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23
```
##### Train The Model
```
python train.py cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights
python scripts/voc_eval.py results/comp4_det_test_
```
mAP test on released models
```
yolo-voc.weights 544 0.7682 (paper: 78.6)
yolo-voc.weights 416 0.7513 (paper: 76.8)
tiny-yolo-voc.weights 416 0.5410 (paper: 57.1)
```
