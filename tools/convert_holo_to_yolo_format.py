import json
import os
import numpy as np

from tqdm import tqdm
from pycocotools.coco import COCO

names = [
    'traffic sign',
    'traffic light'
]


def convert(img_size, anns):
    dw = 1./(img_size[0])
    dh = 1./(img_size[1])

    cate = anns['category_id']
    box = anns['bbox']
    kpts = anns['keypoints']

    np_kpts = np.array(kpts, dtype=float).reshape((-1, 3))
    cx = np_kpts[:, 0].mean()
    cy = np_kpts[:, 1].mean()
    max_x = np_kpts[:, 0].max()
    max_y = np_kpts[:, 1].max()
    min_x = np_kpts[:, 0].min()
    min_y = np_kpts[:, 1].min()
    w = np.clip(max_x - min_x, 1, 1e5)
    h = np.clip(max_y - min_y, 1, 1e5)
    if w < 1 or h < 1:
        print()
    cx = cx * dw
    cy = cy * dh
    w = w * dw
    h = h * dh

    xs = map(lambda x: x * dw, kpts[0::3])
    ys = map(lambda x: x * dh, kpts[1::3])
    xys = []
    for i, j in zip(xs, ys):
        xys.extend([i, j])
    if cate == 2:  # traffic light
        return [cx, cy, w, h] + xys + [cx, cy] * 2
    return [cx, cy, w, h] + xys


# Object Instance 类型的标注
json_file = '/media/holo/C022AA4B225A6D42/data/20201125/annotations/train_sign_led_512x256_modi.json'
coco = COCO(json_file)  # 加载解析标注文件
data = json.load(open(json_file, 'r'))  # json文件

# 保存的路径
ana_txt_save_dir = '/media/holo/C022AA4B225A6D42/data/20201125/yolo_format/train/kpt_labels'
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
        box = convert((img_width, img_height), ann)
        if box[2] < 1e-5 or box[3] < 1e-5:
            print(box)
        old_category_id = ann['category_id']
        cat = coco.loadCats(old_category_id)[
            0]['name']  # 获取该COCO Cat ID对应的物体种类名
        new_category_id = names.index(cat)
        if new_category_id > 80:
            print(cat)
        f_txt.write("%s %s %s %s %s %s %s %s %s %s %s %s %s\n" %
                    (new_category_id, box[0], box[1], box[2], box[3],
                     box[4], box[5], box[6], box[7], box[8], box[9], box[10], box[11]))
    f_txt.close()
