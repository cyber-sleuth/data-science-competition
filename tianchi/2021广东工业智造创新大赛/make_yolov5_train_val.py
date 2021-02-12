import os
import shutil
import random
from tqdm import tqdm

ratio = 0.1
img_dir = './slice/JPEGImages'
label_dir = './slice/worktxt'

train_img_dir = './tile/images/train'
val_img_dir = './tile/images/val'
train_label_dir = './tile/labels/train'
val_label_dir = './tile/labels/val'

if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
if not os.path.exists(val_label_dir):
    os.makedirs(val_label_dir)

names = os.listdir(img_dir)

val_names = random.sample(names, int(len(names) * ratio))

cnt_1 = 0
cnt_2 = 0
for name in tqdm(names):
    if name in val_names:

        shutil.copy(os.path.join(img_dir, name), os.path.join(val_img_dir, name))
        shutil.copy(os.path.join(label_dir, name[:-4] + '.txt'), os.path.join(val_label_dir, name[:-4] + '.txt'))
    else:

        shutil.copy(os.path.join(img_dir, name), os.path.join(train_img_dir, name))
        shutil.copy(os.path.join(label_dir, name[:-4] + '.txt'), os.path.join(train_label_dir, name[:-4] + '.txt'))
