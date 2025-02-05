import os.path

import numpy as np
from tqdm import trange
import pascal_dataloader
from evaluate_detection.box_ops import to_rectangle
from evaluate_detection.canvas_ds import CanvasDataset
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *



def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../images/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='/data1/dataset/', help='pascal base dir')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--dataset_type', default='pascal',
                        choices=['pascal', 'pascal_det'])
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)

    parser.add_argument('--shots', default=16, type=int)
    parser.add_argument('--random_num', default=0, type=int)
    parser.add_argument('--choose_num', default=0, type=int)
    parser.add_argument('--trn', action='store_true')
    parser.set_defaults(trn=False)
    parser.add_argument('--use_class', action='store_true')
    parser.set_defaults(use_class=False)
    parser.add_argument('--method', default='sum')

    parser.add_argument('--reward', action='store_true')
    parser.set_defaults(reward=False)

    parser.add_argument('--sim', action='store_true')
    parser.set_defaults(sim=False)

    parser.add_argument('--supsim', action='store_true')
    parser.set_defaults(supsim=False)

    parser.add_argument('--aug', action='store_true')
    parser.set_defaults(aug=False)

    # shots: int = 4, use_class: bool = False, random_num: int = 0, choose_num: int = 0, trn: bool = False
    return parser


# def _generate_result_for_ens(args, model, canvases, method='sum'):
#     ids_shuffle, len_keep = generate_mask_for_evaluation()
#     num_patches = 14
#     if method == 'sum':
#         canvas, canvas2 = canvases[0], canvases[1]
#         mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
#         _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
#         x1 = torch.softmax(x1, dim=-1)
#         x2 = torch.softmax(x2, dim=-1)
#         y = ((x1 + x2)/2).argmax(dim=-1)
#         im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)
#
#     elif method == 'sum_pre':
#         canvas, canvas2 = canvases[0], canvases[1]
#         mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
#         _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
#         y = ((x1 + x2) / 2).argmax(dim=-1)
#         im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)
#
#     elif method == 'mult':
#         canvas, canvas2 = canvases[0], canvases[1]
#         mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
#         _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
#         x1 = torch.softmax(x1, dim=-1)
#         x2 = torch.softmax(x2, dim=-1)
#         y = (x1 * x2).argmax(dim=-1)
#         im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)
#
#     elif method == 'max':
#         canvas, canvas2 = canvases[0], canvases[1]
#         mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
#         _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
#         x1 = torch.softmax(x1, dim=-1)
#         x2 = torch.softmax(x2, dim=-1)
#         y = torch.argmax(torch.max(torch.stack([x1,x2], dim=-1), dim=-1)[0], dim=-1)
#         im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)
#
#     # elif method == 'union':
#     #     canvas, canvas2 = canvases[0], canvases[1]
#     #     mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
#     #     _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
#     #     y1 = torch.argmax(x1, dim=-1)
#     #     y2 = torch.argmax(x2, dim=-1)
#     #     im_paste1, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y1)
#     #     im_paste2, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y2)
#     #
#     #
#     else:
#         raise ValueError("Wrong ens")
#     canvas = torch.einsum('chw->hwc', canvas)
#     canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
#
#     return np.uint8(canvas), np.uint8(im_paste[0]), mask

def _generate_result_for_ens(args, model, canvases, method='sum',reward_list=[],now_type=""):
    if now_type == "":
        ids_shuffle, len_keep = generate_mask_for_evaluation()
    else:
        ids_shuffle, len_keep = generate_mask_for_evaluation_aug(now_type)
    num_patches = 14
    if method == 'sum':
        x1_list = []
        for i in range(len(canvases)):
            canvas = canvases[i]
            mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
            if len(reward_list) != 0:
                x1_list.append(torch.softmax(x1, dim=-1)*reward_list[i])
            else:
                x1_list.append(torch.softmax(x1, dim=-1))
        y = torch.mean(torch.stack(x1_list,dim=0),dim=0).argmax(dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    elif method == 'sum_pre':
        x1_list = []
        for i in range(len(canvases)):
            canvas = canvases[i]
            mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model,
                                                           canvas.unsqueeze(0))
            if len(reward_list) != 0:
                x1_list.append(x1*reward_list[i])
            else:
                x1_list.append(x1)
        y = torch.mean(torch.stack(x1_list, dim=0), dim=0).argmax(dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    elif method == 'mult':

        mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model,
                                                       canvases[0].unsqueeze(0))
        x1 = torch.softmax(x1, dim=-1)
        for i in range(len(canvases))[1:]:
            canvas = canvases[i]
            _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model,
                                                           canvas.unsqueeze(0))
            x2 = torch.softmax(x2, dim=-1)
            x1 = x1 * x2
        if len(reward_list) != 0:
            for i in range(len(canvases)):
                x1 = x1 *reward_list[i]
        y = x1.argmax(dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    elif method == 'max':
        x1_list = []
        for i in range(len(canvases)):
            canvas = canvases[i]
            mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model,
                                                           canvas.unsqueeze(0))
            if len(reward_list) != 0:
                x1_list.append(x1*reward_list[i])
            else:
                x1_list.append(x1)
        y = torch.max(torch.stack(x1_list, dim=0), dim=0)[0].argmax(dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    # elif method == 'union':
    #     canvas, canvas2 = canvases[0], canvases[1]
    #     mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
    #     _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
    #     y1 = torch.argmax(x1, dim=-1)
    #     y2 = torch.argmax(x2, dim=-1)
    #     im_paste1, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y1)
    #     im_paste2, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y2)
    #
    #
    else:
        raise ValueError("Wrong ens")
    for i in range(len(canvases)):
        canvases[i] = torch.einsum('chw->hwc', canvases[i])
        canvases[i] = np.uint8(torch.clip((canvases[i].cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy())
    return canvases, np.uint8(im_paste[0])

def _generate_result_for_canvas(args, model, canvas,now_type=""):
    """canvas is already in the right range."""
    if now_type == "":
        ids_shuffle, len_keep = generate_mask_for_evaluation()
    else:
        ids_shuffle, len_keep = generate_mask_for_evaluation_aug(now_type)
    _, im_paste, _ = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device)
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)

@torch.no_grad()
def evaluate(args):
    type_list = ['rd','ld','lu','ru','ru','lu','ld','rd']
    if args.reward:
        # # dev_list = ['2009_000763', '2010_003103', '2009_004492', '2008_005945', '2009_001558', '2011_001924']
        # # reward_list = [ 1.69121347,  0.81368194, -1.90031481,  1.89170468, -1.07922453,  2.55991311]
        # # reward_list = [0.33231354, 0.00281658, 0.28620103, 0.69112825, 0.67552068, 0.0152784 ]
        # # dev_list = ['2008_006969', '2009_002620', '2011_002088', '2009_002089', '2008_007261', '2010_001819'] # 10
        # # reward_list = [ 0.18086846,  0.05199669,  0.10348785, -0.20566507,  0.0446233,  -0.18244296]
        # # reward_list = [ 1.12130954,  0.44745984,  0.62416023, -1.75968193,  0.42388743, -1.55458464]
        # # dev_list = ['2008_007261', '2010_002244', '2011_002890', '2009_001494', '2009_001848', '2010_005379'] # 20 s0
        # # reward_list = [ 0.02271529,  0.10932632,  0.36920488, -0.07201656, -0.00408417,  0.17957011]
        # # dev_list = ['2008_005010', '2009_002588', '2011_001653', '2010_002018', '2008_008654', '2009_003901'] # 20 s1
        # reward_list = [-0.60433575,  1.28967107,  4.42667367,  1.69501651,  0.39483314, -1.50573253]
        # dev_list = ['2008_000278', '2008_000940', '2009_001759', '2008_002080', '2008_001602', '2008_006731'] # 20 s2
        # # reward_list = [-0.06344693,  0.24516399, -0.23182482,  0.43393535, -0.11537927,  0.05095833]
        # # reward_list = [ 0.40792241,  1.23742152,  2.68191676, -0.71570719, -0.15431737,  1.43245649]
        # # dev_list = ['2009_001494', '2008_000522', '2008_007261', '2010_000591', '2010_005379', '2009_000130']
        # # reward_list = [-1.57721068,  1.85560808,  0.63728272,  1.51152078,  2.62798839, -0.58797293]
        # # reward_list = [-0.1528296,   0.20493595,  0.05857559,  0.15980379,  0.32729131, -0.05309888] # n = 2
        # # dev_list = ['train_436', 'train_122', 'train_422', 'train_496', 'train_443', 'train_385'] # det shots = 64
        # # reward_list = [ 0.05316872, -0.05329357, -0.028872,   -0.05727021,  0.0704484,   0.        ]

        # dev_list = ['2009_002946', '2011_002560', '2008_006111', '2008_006855', '2010_005093', '2010_003599'] # 20 s0
        # reward_list = [-2.707269,   -3.72935146, -1.17210495, -3.01580859, -1.40209498, -2.261159  ]
        dev_list = ['2008_003060', '2008_005638', '2010_001892', '2010_003970', '2008_000660', '2008_000578'] # 20 s1
        reward_list = [-1.26028824, -1.24814835, -0.79478712, -1.15942645, -3.08715721, -1.49213884]

        reward_list = np.array(reward_list)
        best_list = np.array(dev_list)[reward_list < 0]
        reward_list_new = np.array(reward_list)[reward_list < 0]
        reward_list_new = reward_list_new/np.sum(reward_list_new) * len(reward_list_new)
        if len(best_list) == 0 or len(best_list) == len(reward_list):
            best_list = np.array(dev_list)[reward_list < np.mean(reward_list)]
            reward_list_new = np.array(reward_list)[reward_list < np.mean(reward_list)]
            reward_list_new = 1/(reward_list_new+1e-5)
            reward_list_new = reward_list_new/np.sum(reward_list_new) * len(reward_list_new)

        # best_list = np.array(dev_list)[reward_list < np.mean(reward_list)]
        # reward_list_new = np.array(reward_list)[reward_list < np.mean(reward_list)]
        # reward_list_new = 1/(reward_list_new+1e-5)
        # reward_list_new = reward_list_new/np.sum(reward_list_new) * len(reward_list_new)

        dev_list = best_list.tolist()
        
        reward_list = torch.tensor(reward_list_new)
        assert not args.trn
    else:
        dev_list = []
        # dev_list = ['2011_002132', '2010_003757'] # s0
        # dev_list = ["2010_000576"] #s1
        # dev_list = ["2008_000579"] #s2
        # dev_list = ["2009_003790"] #s3
        # dev_list = ["2010_005274"] #last
        # # s3 seed0
        # dev_list = [['2009_003790'], ['2009_003790'], ['2009_003790'], ['2009_003790'], ['2009_003790'], ['2009_003790'], ['2009_003790'], ['2009_003790'], ['2009_003790'], ['2009_003790'], ['2010_005274'], ['2009_005165'], ['2011_002810'], ['2009_005165'], ['2010_001310'], ['2010_005274'], ['2010_001310'], ['2010_002366'], ['2010_001310'], ['2009_005165']]
        # dev_list = ['train_480'] #det
        # dev_list = ['2010_003757'] # s0 top - 1
        # dev_list = ['2009_001343'] #s3 seed1 top1
        # dev_list = [['2010_000233'], ['2010_000233'], ['2010_000233'], ['2010_000233'], ['2010_000233'], ['2010_000233'], ['2010_000233'], ['2010_000233'], ['2010_000233'], ['2010_000233'], ['2008_006389'], ['2009_001343'], ['2008_006389'], ['2009_000379'], ['2009_004180'], ['2009_000379'], ['2009_000379'], ['2008_006389'], ['2008_006389'], ['2009_001343']]
        reward_list = []


    with open(os.path.join(args.output_dir, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    ds = {
        'pascal': pascal_dataloader.DatasetPASCAL,
        'pascal_det': CanvasDataset
    }[args.dataset_type](args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple,
                         shots=args.shots,use_class=args.use_class,random_num=args.random_num,choose_num=args.choose_num,
                         trn=args.trn,dev_list=dev_list,aug=args.aug,sim=args.sim,supsim=args.supsim)
    
    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
        log.write('dev_list:'+str(dev_list) + '\n')
        log.write('reward_list:'+str(reward_list) + '\n')

    # shots: int = 4, use_class: bool = False, random_num: int = 0, choose_num: int = 0, trn: bool = False
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)
    # Build the transforms:
    # eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
    eval_dict = {'iou': [], 'color_blind_iou': [], 'accuracy': [],'query_name':[],'support_name':[]}
    for idx in trange(len(ds)):
        # idx = idx * 6
        # if idx == 4*6:
        #     break
        canvas = ds[idx]['grid']
        support_name = ds[idx]['support_name']
        query_name = ds[idx]['query_name']
        if isinstance(canvas,list) and len(canvas) > 1:
            if args.aug:
                # assert 0
                for i in range(len(canvas)):
                    for j in range(len(canvas[0])):
                        if args.dataset_type != 'pascal_det':
                            canvas[i][j] = (canvas[i][j] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                original_image, generated_result = _generate_result_for_ens(args, model, canvas[:][0],args.method,reward_list)
                if args.purple:
                    generated_result = round_image(generated_result, [YELLOW, PURPLE], t=args.t)
                else:
                    generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)

                generated_result = np.array(generated_result,dtype=np.float32)
                re_list = [[]]
                for i in range(1,8):
                    _, re = _generate_result_for_ens(args, model, canvas[:][i],args.method,reward_list,now_type=type_list[i])
                    if args.purple:
                        re = round_image(re, [YELLOW, PURPLE], t=args.t)
                    else:
                        re = round_image(re, [WHITE, BLACK], t=args.t)
                    re_list.append(np.array(re,dtype=np.float32))
                
                generated_result[113:,113:] += re_list[1][113:,:111] + \
                                                re_list[2][:111,:111] + \
                                                re_list[3][:111,113:] + \
                                                re_list[4][:111,113:] + \
                                                re_list[5][:111,:111] + \
                                                re_list[6][113:,:111] + \
                                                re_list[7][113:,113:]
                generated_result[113:,113:] /= 8

                generated_result = np.array(generated_result,dtype=np.uint8)

            else:
                if args.dataset_type != 'pascal_det':
                    for i in range(len(canvas)):
                        canvas[i] = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                original_image, generated_result = _generate_result_for_ens(args, model, canvas,args.method,reward_list)
            
            for i in range(len(original_image)):
                if args.purple:
                    original_image[i] = round_image(original_image[i], [YELLOW, PURPLE])
                else:
                    original_image[i] = round_image(original_image[i], [WHITE, BLACK])

        else:
            if isinstance(canvas,list) and len(canvas) == 1:
                canvas = canvas[0]
            # print(support_name,query_name,canvas.shape)
            if args.aug:
                for i in range(len(canvas)):
                    if args.dataset_type != 'pascal_det':
                        canvas[i] = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                original_image, generated_result = _generate_result_for_canvas(args, model, canvas[0])


                # Image.fromarray(np.uint8(generated_result)).save(
                #     os.path.join(args.output_dir, f'generated_{idx}_0.png'))
                if args.purple:
                    generated_result = round_image(generated_result, [YELLOW, PURPLE], t=args.t)
                else:
                    generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)


                generated_result = np.array(generated_result,dtype=np.float32)
                re_list = [[]]
                for i in range(1,8):
                    _,re = _generate_result_for_canvas(args, model, canvas[i],now_type=type_list[i])
                    if args.purple:
                        re = round_image(re, [YELLOW, PURPLE], t=args.t)
                    else:
                        re = round_image(re, [WHITE, BLACK], t=args.t)

                    re_list.append(np.array(re,dtype=np.float32))

                    # Image.fromarray(np.uint8(re)).save(
                    # os.path.join(args.output_dir, f'generated_{idx}_{i}.png'))


                generated_result[113:,113:] += re_list[1][113:,:111] + \
                                                re_list[2][:111,:111] + \
                                                re_list[3][:111,113:] + \
                                                re_list[4][:111,113:] + \
                                                re_list[5][:111,:111] + \
                                                re_list[6][113:,:111] + \
                                                re_list[7][113:,113:] 
                # generated_result[113:,113:] = re_list[7][113:,113:] * 8

                generated_result[113:,113:] /= 8

                generated_result = np.array(generated_result,dtype=np.uint8)

                # Image.fromarray(np.uint8(generated_result)).save(
                #     os.path.join(args.output_dir, f'generated_{idx}_aug.png'))
                
            else:
                if args.dataset_type != 'pascal_det':
                    canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                # Calculate the original_image and the result
                original_image, generated_result = _generate_result_for_canvas(args, model, canvas)
            
            if args.purple:
                original_image = round_image(original_image, [YELLOW, PURPLE])
            else:
                original_image = round_image(original_image, [WHITE, BLACK])



        # if args.output_dir:
        #     if isinstance(original_image,list):
        #         for i in range(len(original_image)):
        #             Image.fromarray(np.uint8(original_image[i])).save(
        #                 os.path.join(args.output_dir, f'original_{idx}_{i}.png'))
        #     else:
        #         Image.fromarray(np.uint8(original_image)).save(
        #             os.path.join(args.output_dir, f'original_{idx}.png'))
        #     Image.fromarray(np.uint8(generated_result)).save(
        #         os.path.join(args.output_dir, f'generated_{idx}.png'))
        

        # if args.output_dir:
        #     Image.fromarray(np.uint8(generated_result)).save(
        #         os.path.join(args.output_dir, f'generated_before_rounding_{idx}.png'))

        if args.purple:
            generated_result = round_image(generated_result, [YELLOW, PURPLE], t=args.t)
        else:
            generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)

        # if args.output_dir:
        #     Image.fromarray(np.uint8(generated_result)).save(
        #         os.path.join(args.output_dir, f'generated_rounded_{idx}.png'))

        if args.task == 'detection':
            generated_result = to_rectangle(generated_result)
        
        # if args.output_dir:
        #     if isinstance(original_image,list):
        #         for i in range(len(original_image)):
        #             Image.fromarray(np.uint8(original_image[i])).save(
        #                 os.path.join(args.output_dir, f'original_{idx}_{i}_f.png'))
        #     else:
        #         Image.fromarray(np.uint8(original_image)).save(
        #             os.path.join(args.output_dir, f'original_{idx}_f.png'))
        #     Image.fromarray(np.uint8(generated_result)).save(
        #         os.path.join(args.output_dir, f'generated_{idx}_f.png'))

        # if args.output_dir:
        #     Image.fromarray(np.uint8(original_image)).save(
        #         os.path.join(args.output_dir, f'original_rounded_{idx}.png'))
        #     Image.fromarray(np.uint8(generated_result)).save(
        #         os.path.join(args.output_dir, f'generated_fixed_{idx}.png'))

        # if args.purple:
        #     current_metric = calculate_metric(args, original_image, generated_result, fg_color=YELLOW, bg_color=PURPLE)
        # else:
        #     current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)

        if isinstance(original_image,list) and len(original_image) > 1:
            now_m = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
            for i in range(len(original_image)):
                if args.purple:
                    current_metric = calculate_metric(args, original_image[i], generated_result, fg_color=YELLOW, bg_color=PURPLE)
                else:
                    current_metric = calculate_metric(args, original_image[i], generated_result, fg_color=WHITE, bg_color=BLACK)

                for c_i, j in current_metric.items():
                    now_m[c_i] += j/len(original_image)
            
            for c_i, j in current_metric.items():
                eval_dict[c_i] += [now_m[c_i]]
            
            
        else:
            if isinstance(original_image,list) and len(original_image) == 1:
                original_image = original_image[0]
            if args.purple:
                current_metric = calculate_metric(args, original_image, generated_result, fg_color=YELLOW, bg_color=PURPLE)
            else:
                current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)
            for i, j in current_metric.items():
                eval_dict[i] += [j]



        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
            log.write(str(idx) + '\t' + str(current_metric) + '\n')
        eval_dict['support_name'] += [support_name]
        eval_dict['query_name'] += [query_name]
        
    # with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
    #     log.write('all\t' + str(eval_dict) + '\n')
    np.save(os.path.join(args.output_dir, 'out.npy'), eval_dict, allow_pickle=True)

if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(3)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
