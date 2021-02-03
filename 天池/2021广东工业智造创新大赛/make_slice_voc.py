# -*- coding: utf-8 -*-
"""
@Time ： 2021/1/9 下午6:16
@Auth ： limzero
@File ：make_slice_train_voc.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
# -*- coding: utf-8 -*-
"""
@Time ： 2021/1/8 下午5:46
@Auth ： limzero
@File ：slice_img.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 02:53:01 2016
@author: avanetten
"""

import os
import cv2
import time
import codecs
import xml.etree.ElementTree as ET
from tqdm import tqdm


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def deal_xml(xml_f):
    tree = ET.parse(xml_f)
    root = tree.getroot()
    object_list = []
    # 处理每个标注的检测框
    for obj in get(root, 'object'):
        # 取出检测框类别名称
        category = get_and_check(obj, 'name', 1).text
        # 更新类别ID字典
        bndbox = get_and_check(obj, 'bndbox', 1)
        xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
        ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
        xmax = int(get_and_check(bndbox, 'xmax', 1).text)
        ymax = int(get_and_check(bndbox, 'ymax', 1).text)
        assert (xmax > xmin)
        assert (ymax > ymin)
        o_width = abs(xmax - xmin)
        o_height = abs(ymax - ymin)
        obj_info = [xmin, ymin, xmax, ymax, category]
        object_list.append(obj_info)
    return object_list


def exist_objs(list_1, list_2):
    '''
    list_1:当前slice的图像
    list_2:原图中的所有目标
    return:原图中位于当前slicze中的目标集合
    '''
    return_objs = []
    s_xmin, s_ymin, s_xmax, s_ymax = list_1[0], list_1[1], list_1[2], list_1[3]
    for vv in list_2:
        xmin, ymin, xmax, ymax, category = vv[0], vv[1], vv[2], vv[3], vv[4]
        if s_xmin < xmin < s_xmax and s_ymin < ymin < s_ymax:  # 目标点的左上角在切图区域中
            if s_xmin < xmax < s_xmax and s_ymin < ymax < s_ymax:  # 目标点的右下角在切图区域中
                x_new = xmin - s_xmin
                y_new = ymin - s_ymin
                return_objs.append([x_new, y_new, x_new + (xmax - xmin), y_new + (ymax - ymin), category])

    return return_objs


def make_slice_voc(out_annos_dir, img_name, exiset_obj_list, sliceHeight=1024, sliceWidth=1024):
    """
    out_annos_dir: 输出切片后的annos的目录
    img_name: 切片后的图片名
    exiset_obj_list: 切图中存在的对象列表
    """
    #
    with codecs.open(os.path.join(out_annos_dir, img_name[:-4] + '.xml'), 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<filename>' + img_name + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(sliceWidth) + '</width>\n')
        xml.write('\t\t<height>' + str(sliceHeight) + '</height>\n')
        xml.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml.write('\t</size>\n')
        cnt = 0
        for obj in exiset_obj_list:
            #
            bbox = obj[:4]
            class_name = obj[-1]
            xmin, ymin, xmax, ymax = bbox
            #
            xml.write('\t<object>\n')
            xml.write('\t\t<name>' + class_name + '</name>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
            cnt += 1
        assert cnt > 0
        xml.write('</annotation>')


###############################################################################
def slice_im(original_img_path, original_annos_path, sliced_img_dir, sliced_annos_dir,
             sliceHeight=1024, sliceWidth=1024, zero_frac_thresh=0.2, overlap=0.2, verbose=False):
    img_name = os.path.basename(original_img_path)
    # 分离出文件名前缀和后缀
    prefix, postfix = img_name.split('.')

    object_list = deal_xml(original_annos_path)

    image0 = cv2.imread(original_img_path, 1)  # color

    win_h, win_w = image0.shape[:2]
    # print(win_h,win_w)
    # if slice sizes are large than image, pad the edges
    # 避免出现切图的大小比原图还大的情况
    pad = 0
    if sliceHeight > win_h:
        pad = sliceHeight - win_h
    if sliceWidth > win_w:
        pad = max(pad, sliceWidth - win_w)
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=border_color)

    win_size = sliceHeight * sliceWidth

    # t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    for y0 in range(0, image0.shape[0], dy):
        for x0 in range(0, image0.shape[1], dx):
            n_ims += 1
            #
            # 这一步确保了不会出现比要切的图像小的图，其实是通过调整最后的overlop来实现的
            # 举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
            #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
            if y0 + sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0
            #
            slice_xmax = x + sliceWidth
            slice_ymax = y + sliceHeight
            exiset_obj_list = exist_objs([x, y, slice_xmax, slice_ymax], object_list)
            if exiset_obj_list != []:  # 如果不为空,说明切出来的这一张图存在目标
                #
                # sliced image
                window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                # get black and white image
                # 剔除背景，背景几乎为黑色
                window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

                # find threshold that's not black
                ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
                non_zero_counts = cv2.countNonZero(thresh1)
                zero_counts = win_size - non_zero_counts
                zero_frac = float(zero_counts) / win_size
                # print "zero_frac", zero_fra
                # skip if image is mostly empty
                if zero_frac >= zero_frac_thresh:
                    if verbose:
                        print("Zero frac too high at:", zero_frac)
                    continue
                    # else save
                else:
                    out_img_name = f'{prefix}_{y}_{x}_{sliceHeight}x{sliceWidth}_{pad}.{postfix}'
                    outpath = os.path.join(
                        sliced_img_dir, out_img_name)
                    #
                    if verbose:
                        print("outpath:", outpath)
                    cv2.imwrite(outpath, window_c)
                    n_ims_nonull += 1

                    # ------制作新的xml------
                    make_slice_voc(sliced_annos_dir, out_img_name, exiset_obj_list, sliceHeight, sliceWidth)

    # print("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull, \
    # "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    # print("Time to slice", image_path, time.time() - t0, "seconds")


if __name__ == "__main__":
    use_demo = False

    if use_demo:
        image_path = '197_3_t20201119085029342_CAM2.jpg'
        ann_path = '197_3_t20201119085029342_CAM2.xml'
        out_img_dir = './slice/JPEGImages_demo'
        out_annos_dir = './slice/annotations_demo'
        if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)
        if not os.path.exists(out_annos_dir): os.makedirs(out_annos_dir)
        slice_im(image_path, ann_path, out_img_dir, out_annos_dir, sliceHeight=640, sliceWidth=640, verbose=True)

    else:
        raw_images_dir = 'E:/Datasets/tile_round1/tile_round1_train_20201231/train_imgs'  #
        raw_annos_dir = './slice/voc_style_Annotations'  # 上一步convert_
        out_annos_dir = './slice/annotations'  # 切出来的标签也保存为voc格式
        out_imgs_dir = './slice/JPEGImages'
        if not os.path.exists(out_annos_dir): os.makedirs(out_annos_dir)
        if not os.path.exists(out_imgs_dir): os.makedirs(out_imgs_dir)
        cnt = 0
        for per_img_name in tqdm(os.listdir(raw_images_dir)):
            image_path = os.path.join(raw_images_dir, per_img_name)
            ann_path = os.path.join(raw_annos_dir, per_img_name[:-4] + '.xml')
            out_name_prefix = str(cnt)
            slice_im(image_path, ann_path, out_imgs_dir, out_annos_dir, sliceHeight=1024, sliceWidth=1024)
            cnt += 1
