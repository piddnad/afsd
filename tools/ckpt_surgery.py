import torch
import argparse
import os


def ckpt_surgery(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.
    """
    def surgery(param_name, is_weight, tar_size, ckpt):
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)

        # 将base类别参数拷贝到新的分类头参数中
        if args.dataset == 'coco':
            for idx, c in enumerate(BASE_CLASSES):
                if 'cls_score' in param_name:
                    new_weight[IDMAP[c]] = pretrained_weight[idx]
                else:
                    new_weight[IDMAP[c]*4:(IDMAP[c]+1)*4] = \
                        pretrained_weight[idx*4:(idx+1)*4]
        else:
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]

        if args.no_bg:
            print('Ignored bg class weights.')  # Not to copy background class weights during surgery
        else:
            if 'cls_score' in param_name:
                new_weight[-1] = pretrained_weight[-1]  # copy bg class weight
        ckpt['model'][weight_name] = new_weight

    surgery_loop(args, surgery)


def surgery_loop(args, surgery):
    # Load checkpoints
    ckpt = torch.load(args.src)
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0

    # Surgery
    if args.method == 'remove':
        for param_name in args.param_name:
            del ckpt['model'][param_name + '.weight']
            if param_name+'.bias' in ckpt['model']:
                del ckpt['model'][param_name+'.bias']
    elif args.method == 'randinit':
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name, tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            surgery(param_name, True, tar_size, ckpt)  # weight
            surgery(param_name, False, tar_size, ckpt)  # bias
    else:
        raise NotImplementedError

    # Save to file
    save_name = args.tar_name + '_' + \
        ('remove' if args.method == 'remove' else 'surgery') + \
        ('_no_bg' if args.no_bg else '') + '.pth'
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, save_name)
    torch.save(ckpt, save_path)
    print('save changed ckpt to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src', type=str, default='',
                        help='Path to the main checkpoint')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Save directory')
    # Surgery method
    parser.add_argument('--method', choices=['remove', 'randinit'],
                        required=True,
                        help='Surgery method.'
                             'remove = remove the final layer of the base detector.'
                             'randinit = randomly initialize novel weights.')
    # Targets
    parser.add_argument('--param-name', type=str, nargs='+',
                        default=['roi_heads.box_predictor.cls_score',
                                 'roi_heads.box_predictor.bbox_pred'],
                        help='Target parameter names')
    parser.add_argument('--tar-name', type=str, default='model_reset',
                        help='Name of the new ckpt')
    # Dataset
    parser.add_argument('--dataset', type=str, default='coco', choices=['voc', 'coco'])

    # Background class weights
    parser.add_argument('--no-bg', action='store_true',
                        help='Not to copy background class weights during surgery')

    args = parser.parse_args()

    if args.dataset == 'coco':
        NOVEL_CLASSES = [
            1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67,
            72,
        ]
        BASE_CLASSES = [
            8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
        IDMAP = {v: i for i, v in enumerate(ALL_CLASSES)}
        TAR_SIZE = 80
    elif args.dataset == 'voc':
        TAR_SIZE = 20
    else:
        raise NotImplementedError

    ckpt_surgery(args)
