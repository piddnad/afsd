import os
import math
import argparse
import numpy as np
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[10], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')

    for shot in args.shot_list:
        # 获得所有实验的 log.txt 路径 --> file_paths
        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if fname.split('_')[0] != '{}shot'.format(shot):
                continue
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue
            file_paths.append(os.path.join(_dir, 'log.txt'))

        # 获得所有实验的精度结果 --> results
        header, results = [], []
        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            if fid == 0:
                header = lineinfos[-2].strip().split(':')[-1].split(',')

            # -----Deprecated：取最后一次test精度------
            # res_info = lineinfos[-1].strip()
            # res = [float(x) for x in res_info.split(':')[-1].split(',')]

            # -----New: 取该log中最高的nAP精度------
            res_list = []
            for idx, line in enumerate(lineinfos):
                line = line.strip()
                if 'copypaste:' in line and 'Task' not in line and 'AP' not in line:
                    res_list.append([float(x) for x in line.split(':')[-1].split(',')])
                    if 'gfsod' in args.res_dir:
                        if len(res_list) > 10:  # coco
                            res_list.sort(key=lambda x: x[12], reverse=True)
                        else:  # voc
                            res_list.sort(key=lambda x: x[7], reverse=True)
                    else:
                        res_list.sort(key=lambda x: x[0], reverse=True)

            results.append([fid] + res_list[0])

        # 求出每个指标的均值和方差
        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
        results.append(['μ'] + avg[1:])
        results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )

        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()