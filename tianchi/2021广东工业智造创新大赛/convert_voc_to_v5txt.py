'''
2020/6/15,标注文件转换xml转txt（vol to yolo）转完后需添加labels文件，即数字序号对应的标签名。

'''

import xml.etree.ElementTree as ET
from tqdm import tqdm

import os

classes = ['边异常', '角异常', '白色点瑕疵', '浅色块瑕疵', '深色点块瑕疵', '光圈瑕疵']


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    if w >= 1:
        w = 0.99
    if h >= 1:
        h = 0.99
    return x, y, w, h


def convert_annotation(voc_path, txt_path):
    with open(voc_path, "r", encoding='UTF-8') as in_file:
        # print(txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        with open(txt_path, "w+", encoding='UTF-8') as out_file:
            out_file.truncate()
            for obj in root.iter('object'):
                # difficult = obj.find('difficult').text
                cls = obj.find('name').text
                # if cls not in classes or int(difficult)==1:
                # continue
                assert cls in classes
                cls_id = classes.index(cls)
                bndbox = obj.find('bndbox')
                b = (float(bndbox.find('xmin').text), float(bndbox.find('xmax').text), float(bndbox.find('ymin').text),
                     float(bndbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    rootpath = './slice'
    yolo_style_annos_dir = os.path.join(rootpath, 'worktxt')
    if not os.path.exists(yolo_style_annos_dir):
        os.mkdir(yolo_style_annos_dir)
    voc_style_annos_dir = os.path.join(rootpath, 'annotations')
    voc_style_anno_names = os.listdir(voc_style_annos_dir)
    for name in tqdm(voc_style_anno_names):
        assert name[-4:].lower() == '.xml'
        voc_path = os.path.join(voc_style_annos_dir, name)
        voc_name = os.path.basename(voc_path)
        txt_name = voc_name[:-4] + '.txt'
        txt_path = os.path.join(yolo_style_annos_dir, txt_name)
        convert_annotation(voc_path, txt_path)
