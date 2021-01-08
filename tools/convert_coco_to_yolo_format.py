import json
import os
from tqdm import tqdm
from pycocotools.coco import COCO

names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


# Object Instance 类型的标注
for type_name in ['val']:
    json_file = f'/media/holo/C022AA4B225A6D42/data/coco/2017/annotations/instances_{type_name}2017.json'
    coco = COCO(json_file)  # 加载解析标注文件
    data = json.load(open(json_file, 'r'))  # json文件

    # 保存的路径
    ana_txt_save_dir = f'/media/holo/C022AA4B225A6D42/data/yolo_format_coco/{type_name}/labels'
    if not os.path.exists(ana_txt_save_dir):
        os.makedirs(ana_txt_save_dir)

    for img in tqdm(data['images'], total=len(data['images'])):
        # print(img["file_name"])
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        # print(img["height"])
        # print(img["width"])
        img_id = img["id"]
        ana_txt_name = filename.split(".")[0] + ".txt"  # 对应的txt名字，与jpg一致
        # print(ana_txt_name)
        f_txt = open(os.path.join(ana_txt_save_dir, ana_txt_name), 'w')
        annIds = coco.getAnnIds(imgIds=img_id)  # 获取该图片对应的所有COCO物体类别标注ID
        for annId in annIds:
            ann = coco.loadAnns(annId)[0]  # 加载标注信息
            box = convert((img_width, img_height), ann["bbox"])
            old_category_id = ann['category_id']
            cat = coco.loadCats(old_category_id)[
                0]['name']  # 获取该COCO Cat ID对应的物体种类名
            new_category_id = names.index(cat)
            if new_category_id > 80:
                print(cat)
            f_txt.write("%s %s %s %s %s\n" %
                        (new_category_id, box[0], box[1], box[2], box[3]))
        f_txt.close()
