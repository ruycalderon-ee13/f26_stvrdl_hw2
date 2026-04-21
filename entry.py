import torch
import torch.nn as NN
import torch.nn.functional as func
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as tfunc2
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert, generalized_box_iou
import matplotlib.pyplot as plt

import argparse
from enum import Enum
from itertools import zip_longest, chain
import json
import random
import time



class RunMode(str, Enum):
    TRAIN = "train"
    INFER = "infer"

def init_device():
    device_string = "cuda" if torch.cuda.is_available() else\
        "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_string)
    
    return device, device_string 

def parse_cmd():
    parser = argparse.ArgumentParser(
        description="Select run mode"
    )

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        choices=[mode.value for mode in RunMode],
        help="Select which operation you'd like to perform: " + ", ".join(
            [mode.value for mode in RunMode])
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        help="Select image root directory"
    )

    parser.add_argument(
        "--annotation_path",
        type=str,
        help="Select annotation file path, input for training, output for inference"
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Specify model weight file to load for inference"
    )

    return parser.parse_args()

class TestImageDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms

        self.image_paths = sorted(
            self.root_dir.glob("*.png"),
            key=lambda p: int(p.stem)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image_id = int(image_path.stem)

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            if self.transforms is not None:
                image = self.transforms(img)
            else:
                image = img

        target = {
            "image_id": image_id,
            "orig_size": (orig_w, orig_h),
        }

        return image, target

def collate_infer(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def check_annotation(image, target):
    image, target = dataset[index]
    labels = [str(im['category_id']) for im in target]
    bbs = [(x,y,x+w,x+h) for (x,y,w,h) in [im['bbox'] for im in target]]
    im_id = target[0]["image_id"]

    boxes = torch.tensor(bbs, dtype=torch.float)
    image_info = training_dataset.coco.loadImgs(im_id)[0]
    image_path = str(Path("/".join(
        [str(training_im),image_info['file_name']])
        ))
    img = read_image(image_path)

    annotated = draw_bounding_boxes(img, boxes, labels=labels, width=0)
    plt.imshow(annotated.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


class SinePosEncode(NN.Module):
    def __init__(self, hidden_dimension, temperature):
        super().__init__()
        assert(hidden_dimension%4 == 0)
        self.d_model = hidden_dimension
        self.temperature = temperature

    def forward(self, x):
        B,C,H,W = x.shape
        device = x.device
        dtype = x.dtype

        assert(C == self.d_model)

        y_grid = torch.arange(H,device=device, dtype=dtype)\
            .unsqueeze(1).repeat(1,W)
        x_grid = torch.arange(W,device=device, dtype=dtype)\
            .unsqueeze(0).repeat(H,1)
        axis_channels = self.d_model//2
        frequency_count = axis_channels//2
        freq_vector = torch.arange(frequency_count, device=device, dtype=dtype)
        divisors = self.temperature ** (2 * freq_vector / frequency_count)
        x_scaled = x_grid.unsqueeze(-1) / divisors
        y_scaled = y_grid.unsqueeze(-1) / divisors

        x_pos = torch.stack(
            (torch.sin(x_scaled), torch.cos(x_scaled)), dim=-1
        ).flatten(-2)
        y_pos = torch.stack(
            (torch.sin(y_scaled), torch.cos(y_scaled)), dim=-1
        ).flatten(-2)

        pos = torch.cat((y_pos, x_pos), dim=-1)
        pos = pos.permute(2,0,1).unsqueeze(0).repeat(B,1,1,1)
        assert(pos.shape[1] == self.d_model)
        return pos

class DecoderMLP(NN.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_count):
        super().__init__()
        assert(layer_count >= 2)
        
        middle_layers = [NN.Linear(hidden_dim, hidden_dim) for i in range(layer_count-2)]
        linear_layers = [NN.Linear(input_dim, hidden_dim)] + middle_layers + [NN.Linear(hidden_dim, output_dim)]
        activations = [NN.ReLU() for _ in range(layer_count-1)]
        layers = list(chain.from_iterable(zip_longest(linear_layers, activations)))[:-1]

        self.mlp = NN.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class DETRDecoderLayer(NN.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class DETREncoderLayer(NN.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True, norm_first=False):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        q = self.with_pos_embed(src, pos)
        k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(
            q,
            k,
            value=src,
            need_weights=False,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class DETR(NN.Module):
    def __init__(self, class_count, query_count):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.class_count = class_count
        self.query_count = query_count

        self.stem = NN.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.blocks = NN.ModuleList([
            base.layer1,
            base.layer2,
            base.layer3
        ])

        # for p in self.stem.parameters():
        #     p.requires_grad = False
        # for p in self.blocks.parameters():
        #     p.requires_grad = False
        
        #transformer
        #encoder
        self.transformer_conv = NN.Conv2d(1024,256, kernel_size=1, bias=False)
        self.pos_encode = SinePosEncode(256,10000.0)

        self.encoder_layers = NN.ModuleList([
            DETREncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=2024,
                dropout=0.1,
                batch_first=True,
                norm_first=False,
            )
            for _ in range(6)
        ])
        

        self.decoder_layers = NN.ModuleList([
            DETRDecoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=2024,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(6)
        ])

        self.query_content_embed = NN.Embedding(query_count, 256)
        self.query_pos_embed = NN.Embedding(query_count, 256)

        self.class_head = NN.Linear(256,class_count+1)
        self.box_head = DecoderMLP(256,256, 4, 3)

        self.transformer_layers = [
            self.transformer_conv,
            self.encoder_layers,
            self.decoder_layers,
            self.query_content_embed,
            self.query_pos_embed,
            self.class_head,
            self.box_head
            ]


    def forward(self, x):
        x = self.stem(x)
        for layer in self.blocks:
            x = layer(x)

        src = self.transformer_conv(x)
        pos = self.pos_encode(src)

        src = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos.flatten(2).permute(0, 2, 1)

        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, pos=pos_flat)


        B = enc_output.shape[0]

        query_content = self.query_content_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1)


        tgt = query_content
        decoder_outputs = []
        for layer in self.decoder_layers:
            tgt = layer(tgt=tgt, memory=enc_output, pos=pos_flat, query_pos=query_pos)
            decoder_outputs.append(tgt)


        class_preds = [self.class_head(output) for output in decoder_outputs]
        box_preds = [self.box_head(output).sigmoid() for output in decoder_outputs]


        return class_preds, box_preds

    def unfreeze_block_i(self, i:int):
        assert(i >= 0 and i < len(self.blocks))
        for p in self.blocks[i].parameters():
            p.requires_grad = True

    def get_stem_params(self):
        return self.stem.parameters()

    def get_block_params(self):
        return self.blocks.parameters()

    def get_transformer_params(self):
        return chain.from_iterable(layer.parameters() for layer in self.transformer_layers)

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

class Matcher(NN.Module):
    def __init__(self):        
        super().__init__()
        self.class_weight = 1.0
        self.box_weight = 5.0
        self.iou_weight = 2.0

    @torch.no_grad()
    def forward(self, class_pred, box_pred, targets):
        B,Q,C = class_pred.shape
        indices = []

        class_prob_flat = class_pred.flatten(0,1).softmax(-1)
        pred_boxes_cxcywh = box_pred.flatten(0,1)
        pred_boxes_xyxy = box_convert(pred_boxes_cxcywh, out_fmt="xyxy", in_fmt="cxcywh")

        tgt_labels = torch.cat([v["labels"] for v in targets])
        tgt_boxes_xyxy = torch.cat([v["boxes"] for v in targets])
        tgt_boxes_cxcywh = box_convert(tgt_boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")

            
        cost_class = -class_prob_flat[:,tgt_labels]
        cost_bbox = torch.cdist(pred_boxes_cxcywh, tgt_boxes_cxcywh, p=1)
        cost_iou = -generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)

        C = cost_class * self.class_weight + \
            cost_bbox * self.box_weight + \
            cost_iou * self.iou_weight
        
        C = C.view(B, Q, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def get_src_query_map(indices):
    image_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    query_indices = torch.cat([src for (src, _) in indices])
    return image_indices, query_indices

def get_tgt_query_map(indices):
    image_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    query_indices = torch.cat([tgt for (_, tgt) in indices])
    return image_indices, query_indices

def class_loss(class_pred, targets, indices, class_count):
    src_logits = class_pred
    idx = get_src_query_map(indices)

    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], class_count,
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    class_weights = torch.ones(class_count + 1, device=src_logits.device)
    class_weights[-1] = 0.1
    loss = func.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=class_weights)
    return loss

def bbox_loss(box_pred, targets, indices):
    idx = get_src_query_map(indices)

    src_boxes_cxcywh = box_pred[idx]

    tgt_boxes_xyxy = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)],dim=0)
    tgt_boxes_cxcywh = box_convert(tgt_boxes_xyxy, in_fmt="xyxy", out_fmt = "cxcywh")
    num_boxes = max(tgt_boxes_cxcywh.shape[0],1)
    loss = func.l1_loss(src_boxes_cxcywh, tgt_boxes_cxcywh, reduction="none")
    return loss.sum()/num_boxes

def iou_loss(box_pred, targets, indices):
    idx = get_src_query_map(indices)

    src_boxes_cxcywh = box_pred[idx]
    src_boxes_xyxy = box_convert(src_boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
    
    tgt_boxes_xyxy = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)],dim=0)
    
    num_boxes = max(tgt_boxes_xyxy.shape[0],1)

    giou = generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)
    loss = 1 - torch.diag(giou)
    return loss.sum() / num_boxes

def get_pred_rows_batch(
    class_logits,
    boxes_cxcywh,
    class_count,
    image_ids,
    scale,
    score_threshold=0.1,
    top_k=10,
    min_box_size=1.0,
):
    probs = class_logits.softmax(-1)              # [B, Q, C+1]
    fg_probs = probs[..., :class_count]           # exclude no-object
    scores, cls_idx = fg_probs.max(dim=-1)        # [B, Q], [B, Q]
    
    if random.random() > 0.99:
        print("Random batch sample")
        print("batch max fg score:", scores.max().item())
        print("batch mean fg score:", scores.mean().item())
        print("keep@0.05:", (scores >= 0.05).sum().item())
        print("keep@0.10:", (scores >= 0.10).sum().item())
        print("keep@0.30:", (scores >= 0.30).sum().item())

    boxes_xywh = box_convert(
        boxes_cxcywh,
        in_fmt="cxcywh",
        out_fmt="xywh"
    ) * scale                                     # [B, Q, 4]

    rows = []
    B, Q, _ = boxes_xywh.shape

    for b in range(B):
        b_scores = scores[b]
        b_cls = cls_idx[b] + 1                    # back to category_id 1..10
        b_boxes = boxes_xywh[b]

        # threshold first
        keep = b_scores >= score_threshold
        b_scores = b_scores[keep]
        b_cls = b_cls[keep]
        b_boxes = b_boxes[keep]

        if b_scores.numel() == 0:
            continue

        # remove tiny / degenerate boxes
        wh_keep = (b_boxes[:, 2] >= min_box_size) & (b_boxes[:, 3] >= min_box_size)
        b_scores = b_scores[wh_keep]
        b_cls = b_cls[wh_keep]
        b_boxes = b_boxes[wh_keep]

        if b_scores.numel() == 0:
            continue

        # keep only top-k per image
        if b_scores.numel() > top_k:
            top_scores, top_idx = torch.topk(b_scores, k=top_k)
            b_scores = top_scores
            b_cls = b_cls[top_idx]
            b_boxes = b_boxes[top_idx]

        # optional: clamp boxes to valid image region
        img_w = scale[b, 0, 0]
        img_h = scale[b, 0, 1]

        b_boxes[:, 0] = b_boxes[:, 0].clamp(0, img_w)
        b_boxes[:, 1] = b_boxes[:, 1].clamp(0, img_h)
        b_boxes[:, 2] = b_boxes[:, 2].clamp(min=0)
        b_boxes[:, 3] = b_boxes[:, 3].clamp(min=0)

        b_boxes[:, 2] = torch.minimum(b_boxes[:, 2], img_w - b_boxes[:, 0])
        b_boxes[:, 3] = torch.minimum(b_boxes[:, 3], img_h - b_boxes[:, 1])

        for box, score, category_id in zip(b_boxes, b_scores, b_cls):
            rows.append({
                "image_id": int(image_ids[b]),
                "bbox": box.tolist(),
                "score": float(score.item()),
                "category_id": int(category_id.item()),
            })

    return rows

def compute_map(annotations, predictions):
    if len(predictions) == 0:
        print("No predictions were produced; returning None (parsed as mAP score of 0.0)")
        return None

    gt = COCO(annotations)
    dt = gt.loadRes(predictions)
    evaluator = COCOeval(gt, dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator

def gather_all_losses(class_preds, box_preds, targets, matcher, class_count, aux_weight=0.5):
    indices = matcher(class_preds[-1], box_preds[-1], targets)

    main_loss_class = class_loss(class_preds[-1], targets, indices, class_count)
    main_loss_bbox = bbox_loss(box_preds[-1], targets, indices)
    main_loss_IoU = iou_loss(box_preds[-1], targets, indices)
    main_loss = main_loss_class + 5.0*main_loss_bbox + 2.0*main_loss_IoU

    auxiliary_losses = torch.tensor(0.0, device=main_loss.device)
    for class_pred, box_pred in zip(class_preds[:-1], box_preds[:-1]):
        aux_loss_class = class_loss(class_pred, targets, indices, class_count)
        aux_loss_bbox = bbox_loss(box_pred, targets, indices)
        aux_loss_IoU = iou_loss(box_pred, targets, indices)
        aux_loss = aux_loss_class + 5.0*aux_loss_bbox + 2.0*aux_loss_IoU
        auxiliary_losses += aux_loss
    
    auxiliary_losses/=len(class_preds) - 1
    total_loss = main_loss + aux_weight * auxiliary_losses

    if random.random() > 0.99:
        print("Random batch Loss sample")
        print("Total loss:", total_loss.item())
        print("Main loss:", main_loss.item())
        print("Aux loss:", aux_loss.item())

    return total_loss

def train_one_epoch(loader, matcher, model, optimizer, device, orig_image_dim):
    training_loss = 0.0
    counter = 0
    training_out_rows = []
    for image_arr, target_arr in loader:
        if counter%100 == 0:
            print(f"Beginning batch {counter+1}")
        images = torch.stack(image_arr).to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in target_arr]
        H, W = images.shape[-2:]
        scale = torch.tensor([W, H, W, H], dtype=targets[0]["boxes"].dtype, device=device)

        targets = [{
                "boxes": t["boxes"] / scale,
                "labels": t["labels"] - 1,
            } for t in targets
        ]
        optimizer.zero_grad()
        class_preds, box_preds = model(images)
        loss = gather_all_losses(class_preds, box_preds, targets, matcher, model.class_count)
        
        class_pred = class_preds[-1]
        box_pred = box_preds[-1]

        loss.backward()
        optimizer.step()
        training_loss+=loss.item()

        counter+=1
    return training_loss

def validate_one_epoch(loader, matcher, model, optimizer, device, orig_image_dim):
    with torch.no_grad():
        counter = 0
        validate_loss = 0.0
        validate_out_rows = []
        for image_arr, target_arr in loader:
            if counter%100 == 0:
                print(f"Beginning batch {counter+1}")
            images = torch.stack(image_arr).to(device)
            targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in target_arr]
            H, W = images.shape[-2:]
            scale = torch.tensor([W, H, W, H], dtype=targets[0]["boxes"].dtype, device=device)
            
            targets = [{
                    "boxes": t["boxes"] / scale,
                    "labels": t["labels"] - 1,
                } for t in targets
            ]
            class_preds, box_preds = model(images)

            loss = gather_all_losses(class_preds, box_preds, targets, matcher, model.class_count)
            class_pred = class_preds[-1]
            box_pred = box_preds[-1]

            validate_loss+=loss.item()

            image_ids = [t["image_id"] for t in target_arr]
            image_scale = torch.tensor(
                [
                    [
                        orig_image_dim[k][0],
                        orig_image_dim[k][1],
                        orig_image_dim[k][0],
                        orig_image_dim[k][1]
                    ] for k in image_ids
                ],
                dtype=box_pred.dtype,
                device=box_pred.device
            ).unsqueeze(1)
            
            rows = get_pred_rows_batch(
                class_pred,
                box_pred,
                model.class_count,
                image_ids,
                image_scale,
                score_threshold=0.0,
                top_k=model.query_count,
                min_box_size=0.0,
            )
            validate_out_rows.extend(rows)

            counter += 1
        return validate_loss, validate_out_rows
    return None

def eval_one_epoch(loader, matcher, model, device):
    with torch.no_grad():
        counter = 0
        eval_out_rows = []
        for image_arr, target_arr in loader:
            if counter%100 == 0:
                print(f"Beginning batch {counter+1}")
            images = torch.stack(image_arr).to(device)
            class_preds, box_preds = model(images)
            image_ids = [t["image_id"] for t in target_arr]

            image_scale = torch.tensor(
                [
                    [
                        t["orig_size"][0],
                        t["orig_size"][1],
                        t["orig_size"][0],
                        t["orig_size"][1],
                    ]
                    for t in target_arr
                ],
                dtype=box_preds[-1].dtype,
                device=box_preds[-1].device,
            ).unsqueeze(1)
            
            rows = get_pred_rows_batch(
                class_preds[-1],
                box_preds[-1],
                model.class_count,
                image_ids,
                image_scale,
                score_threshold=0.05,
                top_k=15,
                min_box_size=2.0,
            )
            eval_out_rows.extend(rows)

            counter += 1
        return eval_out_rows
    return None


if __name__ == '__main__':
    args = parse_cmd()

    device, device_string = init_device()
    print(f"Using {device_string} backend ")

    model = DETR(class_count=10, query_count=50).to(device)
    
    assert(model is not None)

    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {args.model_path}")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transforms = T.Compose([
        T.ToImage(),
        T.RandomPhotometricDistort(p=0.1),
        T.RandomAffine(
            degrees=3,
            translate=(0.03, 0.03),
            scale=(0.93, 1.03),
            interpolation=InterpolationMode.BILINEAR,
        ),
        T.Resize((384,384), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.SanitizeBoundingBoxes(),
    ])

    deterministic_transforms = T.Compose([
        T.ToImage(),
        T.Resize((384,384), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.SanitizeBoundingBoxes(),
    ])

    eval_transforms = T.Compose([
        T.ToImage(),
        T.Resize((384,384), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    requested_batch_size = 20
    if args.mode == RunMode.TRAIN:
        training_ann = Path("/".join([args.annotation_path, "train.json"]))
        validate_ann = Path("/".join([args.annotation_path, "valid.json"]))
        training_im = Path("/".join([args.root_dir, "train"]))
        validate_im = Path("/".join([args.root_dir, "valid"]))
        

        print([str(p) for p in [training_ann,validate_ann, training_im, validate_im]])
        training_dataset = CocoDetection( 
            root=str(training_im), 
            annFile=str(training_ann), 
            transforms=train_transforms)
        training_image_size_map = {
            image_id: (img_info["width"], img_info["height"])
            for image_id, img_info in training_dataset.coco.imgs.items()
        }
        training_dataset = wrap_dataset_for_transforms_v2(
            training_dataset,
            target_keys=("image_id", "boxes", "labels")
            )
        
        validate_dataset = CocoDetection( 
            root=str(validate_im), 
            annFile=str(validate_ann), 
            transforms=deterministic_transforms)
        validate_image_size_map = {
            image_id: (img_info["width"], img_info["height"])
            for image_id, img_info in validate_dataset.coco.imgs.items()
        }
        validate_dataset = wrap_dataset_for_transforms_v2(
            validate_dataset,
            target_keys=("image_id", "boxes", "labels")
            )

        optimizer = torch.optim.AdamW([
            {"params": model.get_stem_params(), "lr": 1e-5},
            {"params": model.get_block_params(), "lr": 1e-5},
            {"params": model.get_transformer_params(), "lr": 1e-4},
        ])

        loaders = {
            "training":DataLoader(
                training_dataset, 
                batch_size = requested_batch_size, 
                shuffle = True,
                collate_fn = collate),
            "validate":DataLoader(
                validate_dataset, 
                batch_size = requested_batch_size, 
                shuffle = False,
                collate_fn = collate),
        }
    else:
        inference_im = Path("/".join([args.root_dir, "test"]))
        inference_pred = Path("/".join([args.root_dir, "pred.json"]))
        inference_dataset = TestImageDataset( 
            root_dir=str(inference_im),
            transforms=eval_transforms)

        loaders = {
            "evaluate":DataLoader(
                inference_dataset, 
                batch_size = requested_batch_size, 
                shuffle = False,
                collate_fn = collate_infer)
        }

    matcher = Matcher()
    num_epochs = 25
    epoch_data = []
    if args.mode == RunMode.TRAIN:
        print(f"Beginning training with {str(num_epochs)} epochs")
        for epoch in range(num_epochs):
            start = time.perf_counter()
            print(f"Beginning epoch {epoch + 1}")
            model.train()
            print(f"Training in {len(loaders['training'])} batches")

            training_loss = train_one_epoch(
                loaders['training'], matcher, model, optimizer, device, training_image_size_map
            )
            
            training_loss /= len(loaders["training"])

            print("Training Complete. Begin Validate")
            model.eval()
            print(f"Validate in {len(loaders['validate'])} batches")
            validate_loss, validate_pred = validate_one_epoch(
                loaders['validate'], matcher, model, optimizer, device, validate_image_size_map
            )
            print(len(validate_pred))
            validate_loss /= len(loaders["validate"])
            val_eval = compute_map(validate_ann, validate_pred)
            if val_eval:
                val_map_score = val_eval.stats[0]
            else:
                val_map_score = 0.0
            print("Validation Complete.")

            print(f"End Epoch {epoch + 1} of {num_epochs}")
            print(f"Training Loss: {training_loss:0.3f}\t Validate Loss: {validate_loss:0.3f}\t mAP score: {val_map_score:0.3}")
            epoch_data.append((
                round(training_loss,3), 
                round(validate_loss,3),
                round(val_map_score,3)))
            end = time.perf_counter()
            print(f"Elapsed time: {end - start:.4f} seconds")
            torch.save(model.state_dict(),f"tmp_weights{str(epoch)}.pt")
    if args.mode == RunMode.INFER:
        model.eval()
        predictions = eval_one_epoch(
            loaders['evaluate'], matcher, model, device
        )

        with open(str(inference_pred), "w", encoding="utf-8") as f:
            json.dump(predictions, f)

        
        
        #check_annotation(0, training_dataset)
    