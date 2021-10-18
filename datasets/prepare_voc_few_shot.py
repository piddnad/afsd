import numpy as np
from fvcore.common.file_io import PathManager

import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 20],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data = []
    data_per_cat = {c: [] for c in VOC_CLASSES}

    # data数据保存所有file_id
    for year in [2007, 2012]:
        data_file = f'datasets/VOC{year}/ImageSets/Main/trainval.txt'
        with PathManager.open(data_file) as f:
            file_ids = np.loadtxt(f, dtype=np.str).tolist()
        data += file_ids

    # data_per_cat字典内容：{class: [含该class的所有annotation files]}
    for file_id in data:
        year = "2012" if "_" in file_id else "2007"
        dirname = os.path.join("datasets", f"VOC{year}")
        anno_file = os.path.join(dirname, "Annotations", f"{file_id}.xml")
        tree = ET.parse(anno_file)
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(cls)
        for cls in set(clses):
            data_per_cat[cls].append(anno_file)

    # result保存few-shot dict，内容：{class: {shot: 含shot数目个class实例的多个image_name}}
    result = {cls: {} for cls in data_per_cat.keys()}
    shots = [1, 2, 3, 5, 10]
    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for cls in data_per_cat.keys():
            cls_data = []
            for j, shot in enumerate(shots):
                # 每次只用sample和前一shot相比增加的数量的样本，即diff_shot个新样本
                diff_shot = shots[j] - shots[j-1] if j != 0 else 1
                shots_cls = random.sample(data_per_cat[cls], diff_shot)
                num_objs = 0
                for anno_file in shots_cls:
                    tree = ET.parse(anno_file)
                    file = tree.find("filename").text
                    year = tree.find("folder").text
                    name = f'datasets/{year}/JPEGImages/{file}'
                    # if name in cls_data:
                    #     print(f'Duplicate file: seed{i} {cls} {shot}shot {name}')
                    # if name not in cls_data:
                    cls_data.append(name)
                    for obj in tree.findall("object"):
                        if obj.find("name").text == cls:
                            num_objs += 1
                    if num_objs >= diff_shot:
                        break
                # if num_objs < diff_shot:
                #     print(f'* Seed{i} {cls} [{shot}] shots required, but [{shots[j-1] + num_objs}] shots in fact.')
                result[cls][shot] = copy.deepcopy(cls_data)
        save_path = f'datasets/vocsplit/seed{i}'
        os.makedirs(save_path, exist_ok=True)
        for cls in result.keys():
            for shot in result[cls].keys():
                filename = f'box_{shot}shot_{cls}_train.txt'
                with open(os.path.join(save_path, filename), 'w') as fp:
                    fp.write('\n'.join(result[cls][shot]) + '\n')


if __name__ == '__main__':
    args = parse_args()
    generate_seeds(args)
