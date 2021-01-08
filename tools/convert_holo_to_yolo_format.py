import json
import os
from tqdm import tqdm
from pycocotools.coco import COCO

names = [
    'traffic sign',
    'traffic light'
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
json_file = '/media/holo/C022AA4B225A6D42/data/20201125/annotations/test_sign_led_512x256_modi.json'
coco = COCO(json_file)  # 加载解析标注文件
data = json.load(open(json_file, 'r'))  # json文件

# 保存的路径
ana_txt_save_dir = '/media/holo/C022AA4B225A6D42/data/20201125/yolo_format/test/labels'
if not os.path.exists(ana_txt_save_dir):
    os.makedirs(ana_txt_save_dir)

for img in tqdm(data['images'], total=len(data['images'])):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    ana_txt_name = filename.split(".")[0] + ".txt"  # 对应的txt名字，与jpg一致
    # print(ana_txt_name)
    f_txt = open(os.path.join(ana_txt_save_dir, ana_txt_name), 'w')
    annIds = coco.getAnnIds(imgIds=img_id)  # 获取该图片对应的所有COCO物体类别标注ID
    for annId in annIds:
        ann = coco.loadAnns(annId)[0]  # 加载标注信息
        box = convert((img_width, img_height), ann["bbox"])
        if box[2] < 1e-5 or box[3] < 1e-5:
            print(box) 
        old_category_id = ann['category_id']
        cat = coco.loadCats(old_category_id)[
            0]['name']  # 获取该COCO Cat ID对应的物体种类名
        new_category_id = names.index(cat)
        if new_category_id > 80:
            print(cat)
        f_txt.write("%s %s %s %s %s\n" %
                    (new_category_id, box[0], box[1], box[2], box[3]))
    f_txt.close()
