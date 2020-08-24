from pycocotools.coco import COCO
import skimage.io as io
from itertools import combinations
from os.path  import isdir, isfile
from os import mkdir
from shutil import copy

names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop", "sign", "parking", "meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports", "ball", "kite", "baseball", "bat", "baseball", "glove", "skateboard", "surfboard", "tennis", "racket", "bottle", "wine", "glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot", "dog", "pizza", "donut", "cake", "chair", "couch", "potted", "plant", "bed", "dining", "table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell", "phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy", "bear", "hair", "drier", "toothbrush"]

new_id = [0, -1, 1, 2, -1, 3, -1, 4]

dataDir='./data/'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

outpath = "./"+dataType+"_set"

if not isdir(outpath):
    mkdir(outpath)

clss = ['person', 'car','truck', 'motorcycle', 'bus']
combs = []
i = len(clss)
while i>0:
    co = list(combinations(clss, i))
    for c in co:
        combs.append(c)
    i-=1


dataset = set()
for c in combs:
    catIds = coco.getCatIds(catNms=list(c));
    imgIds = coco.getImgIds(catIds=catIds)
    # print(c, len(imgIds))
    for img in imgIds:
        dataset.add(img)

stat = {}

for c in clss:
    stat[c] = 0

i = 0
lend = len(dataset)
ready_l = 0
loader_size = 100


clssIds = coco.getCatIds(catNms=clss);
for data in list(dataset):
    ready = int(loader_size*i/lend)
    if ready > ready_l:
        print("\r [" ,"="*(ready+1), " "*(loader_size-ready-1), end="]")
        ready_l = ready
    img = coco.loadImgs([data])
    annIds = coco.getAnnIds(imgIds=img[0]['id'], catIds=clssIds)
    ann = coco.loadAnns(annIds)
    iw = img[0]['width']
    ih = img[0]['height']
    # print(img)
    f = open("{}/{}.txt".format(outpath, img[0]['file_name'].split(".")[0]), "w+")
    for a in ann:
        x, y, w, h = a['bbox']
        if new_id[a['category_id']-1] > -1:
            if not isfile("{}/{}".format(outpath, img[0]['file_name'])):
                copy("{}/{}/{}".format(dataDir, dataType, img[0]['file_name']), outpath)
            f.write("{} {} {} {} {}\n".format(new_id[a["category_id"]-1], (x+w/2)/iw, (y+h/2)/ih, w/iw, h/ih))
        stat[names[a['category_id']-1]]+=1
    f.close()
    i+=1

print("\nTotal images in dataset : ", len(dataset))
print("Detections by categories: ", stat)