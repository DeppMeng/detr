import math
import argparse
from PIL import Image
import requests
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
from models import build_vis_model

cmap = plt.get_cmap("Dark2")


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--output_layer', default=-1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # custom object query parameters
    parser.add_argument('--sine_query_embed', action='store_true')
    return parser


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush', 'non-object'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, save_name, layer_id):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    count = 0
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=cmap(count), linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(0, count * 15 + 15, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.0), color=cmap(count))
        
        count = count + 1

    plt.axis('off')
    plt.savefig('vis/idx{}_layer{}.png'.format(save_name, layer_id), format='png')

# model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

# img_id = '000000039769'
id_list = [
    '000000000139',
    '000000000285',
    '000000000632',
    '000000000724',
    '000000000776',
    '000000000785',
    '000000000802',
    '000000000872',
    '000000000885',
    '000000001000',
    '000000001268',
    '000000001296',
    '000000001353',
    '000000001425',
    '000000001490',
    '000000001503',
    '000000001532',
    '000000001584',
]


model, _, _ = build_vis_model(args)
# checkpoint = torch.hub.load_state_dict_from_url(
#     args.resume, map_location='cpu', check_hash=True)
# checkpoint = torch.load(args.resume, map_location='cpu')
# model.load_state_dict(checkpoint['model'], strict=False)
model.eval();

idd_list = [[43, 61, 98],
            [71, 92]]

trans_matrix = model.object_trans.weight

for idxx, img_id in enumerate(id_list):
    # img_id = '000000000139'
    url = 'http://images.cocodataset.org/val2017/{}.jpg'.format(img_id)
    im = Image.open(requests.get(url, stream=True).raw)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)


    conv_features, enc_attn_weights, dec_attn_weights, dec_self_atten_weights = [], [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[0].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[0].self_attn.register_forward_hook(
            lambda self, input, output: dec_self_atten_weights.append(output[1])
        ),
    ]

    output_layer = 0
    # propagate through the model
    outputs = model(img, output_layer=output_layer)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0,:,:]

    probas_for_select = outputs['pred_logits'].softmax(-1)[0,:,:-1]
    keep = probas_for_select.max(-1).values > 0.5

    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
    dec_self_atten_weights = dec_self_atten_weights[0]
    
    for hook in hooks:
        hook.remove()

    for count in range(20):
        idd_list = [5 * count, 5 * count + 1, 5 * count + 2, 5 * count + 3, 5 * count + 4]

        keep = torch.zeros(100, dtype=torch.bool)
        keep[idd_list] = True
        print(keep)

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        # plot_results(im, probas[keep], bboxes_scaled, img_id, output_layer)



        print(dec_self_atten_weights)
        # print(dec_attn_weights.shape)
        
        h, w = conv_features['0'].tensors.shape[-2:]

        fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
        colors = COLORS * 100
        
        enc_pos = model.pos[0].view(256, h * w)
        print(enc_pos.shape)
        obj_embed = model.query_embed.weight
        print(obj_embed.shape)
        att_weights = torch.matmul(obj_embed, enc_pos).unsqueeze(0)
        print(att_weights.shape)

        # if count == 0:
        sum_argmax_attn = torch.zeros(100, 10, 10)
        for subcount in range(100):
            # sum_argmax_attn[subcount, :, :] = torch.floor(att_weights[0, subcount].view(h, w) / torch.max(att_weights[0, subcount].view(h, w)))
            sum_argmax_attn[subcount, :, :] = trans_matrix[subcount, :].view(10, 10)
        avg_dec_attn_weights = torch.sum(dec_attn_weights, dim=1, keepdim=True)
        print(avg_dec_attn_weights.shape)
        fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
        colors = COLORS * 100
        for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
            ax = ax_i[0]
            ax.imshow(sum_argmax_attn[idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: all')
            # ax = ax_i[1]
            # ax.imshow(im)
            # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
            #                         fill=False, color='blue', linewidth=3))
            # ax.axis('off')
            ax.set_title(CLASSES[probas[idx].argmax()])
        fig.tight_layout()

        plt.savefig('vis_attn_v4_sineobjquery_trans/idx{}_objectquerysineV4_split_{}.png'.format(img_id, count), format='png')