import torch
import torch.nn as NN
import torch.nn.functional as func
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert, generalized_box_iou
import matplotlib.pyplot as plt

import argparse
from enum import Enum
from itertools import zip_longest, chain



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

    return parser.parse_args()

#confirm 1 is 0, 10 is 9
def check_annotation(index, dataset):
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
            base.layer3,
            base.layer4
        ])

        # for p in self.stem.parameters():
        #     p.requires_grad = False
        # for p in self.blocks.parameters():
        #     p.requires_grad = False
        
        #transformer
        #encoder
        self.transformer_conv = NN.Conv2d(base.fc.in_features,256, kernel_size=1, bias=False)
        self.pos_encode = SinePosEncode(256,10000.0)

        self.encoder_layer = NN.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            norm_first=False
        )

        self.encoder = NN.TransformerEncoder(
            self.encoder_layer,
            num_layers=6
        )

        #decoder
        self.decoder_layer = NN.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.decoder = NN.TransformerDecoder(
            self.decoder_layer,
            num_layers=6
        )

        self.query_embed = NN.Embedding(query_count, 256)
        self.class_head = NN.Linear(256,class_count+1)
        self.box_head = DecoderMLP(256,256, 4, 3)

        self.transformer_layers = [
            self.transformer_conv,
            self.encoder,
            self.decoder,
            self.query_embed,
            self.class_head,
            self.box_head
            ]


    def forward(self, x):
        x = self.stem(x)
        for layer in self.blocks:
            x = layer(x)

        src = self.transformer_conv(x)
        pos = self.pos_encode(src)

        src = src + pos
        src = src.flatten(2).permute(0,2,1)

        enc_output = self.encoder(src)
        B = enc_output.shape[0]

        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        dec_output = self.decoder(tgt=query_pos, memory=enc_output)

        class_pred = self.class_head(dec_output)
        box_pred = self.box_head(dec_output).sigmoid()

        return class_pred, box_pred

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
        self.box_weight = 1.0
        self.iou_weight = 1.0

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

if __name__ == '__main__':
    args = parse_cmd()

    device, device_string = init_device()
    print(f"Using {device_string} backend ")

    model = DETR(class_count=10, query_count=50).to(device)
    
    assert(model is not None)

    if args.mode == RunMode.TRAIN:
        training_ann = Path("/".join([args.annotation_path, "train.json"]))
        validate_ann = Path("/".join([args.annotation_path, "valid.json"]))
        training_im = Path("/".join([args.root_dir, "train"]))
        validate_im = Path("/".join([args.root_dir, "valid"]))
    else:
        inference_im = Path("/".join([args.root_dir, "test"]))
        
    basic_transforms = T.Compose([
        T.ToImage(),
        T.Resize((224, 224)),
        T.ToDtype(torch.float32, scale=True),
        T.SanitizeBoundingBoxes()
    ])

    print([str(p) for p in [training_ann,validate_ann, training_im, validate_im]])
    training_dataset = CocoDetection( 
        root=str(training_im), 
        annFile=str(training_ann), 
        transforms=basic_transforms)
    training_dataset = wrap_dataset_for_transforms_v2(
        training_dataset,
        target_keys=("image_id", "boxes", "labels")
        )
    print(training_dataset)

    validate_dataset = CocoDetection( 
        root=str(validate_im), 
        annFile=str(validate_ann), 
        transforms=basic_transforms)
    validate_dataset = wrap_dataset_for_transforms_v2(
        validate_dataset,
        target_keys=("image_id", "boxes", "labels")
        )

    optimizer = torch.optim.AdamW([
        {"params": model.get_stem_params(), "lr": 1e-5},
        {"params": model.get_block_params(), "lr": 1e-5},
        {"params": model.get_transformer_params(), "lr": 1e-4},
    ])

    requested_batch_size = 50
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

    matcher = Matcher()
    num_epochs = 10
    epoch_data = []
    print(f"Beginning training with {str(num_epochs)} epochs")
    for epoch in range(num_epochs):
        print(f"Beginning epoch {epoch + 1}")
        model.train()
        training_loss = 0.0
        print(f"Training in {len(loaders['training'])} batches")
        counter = 0

        for image_arr, target_arr in loaders['training']:
            if counter%100 == 0:
                print(f"Beginning batch {counter+1}")
            images = torch.stack(image_arr).to(device)
            targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in target_arr]
            H, W = images.shape[-2:]
            scale = torch.tensor([W, H, W, H], dtype=targets[0]["boxes"].dtype, device=device)

            targets = [{
                    "boxes": t["boxes"] / scale,
                    "labels": t["labels"],
                } for t in targets
            ]
            optimizer.zero_grad()
            class_pred, box_pred = model(images)

            indices = matcher(class_pred, box_pred, targets)

            loss_class = class_loss(class_pred, targets, indices, model.class_count)
            loss_bbox = bbox_loss(box_pred, targets, indices)
            loss_IoU = iou_loss(box_pred, targets, indices)
            loss = loss_class + loss_bbox + loss_IoU


            loss.backward()
            optimizer.step()
            training_loss+=loss.item()
            
            counter+=1
        training_loss /= len(loaders["training"])

        print("Training Complete. Begin Validate")
        model.eval()
        validate_loss = 0.0

        print(f"Validate in {len(loaders['validate'])} batches")
        with torch.no_grad():
            counter = 0
            for image_arr, target_arr in loaders['validate']:
                if counter%100 == 0:
                    print(f"Beginning batch {counter+1}")
                images = torch.stack(image_arr).to(device)
                targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in target_arr]
                H, W = images.shape[-2:]
                scale = torch.tensor([W, H, W, H], dtype=targets[0]["boxes"].dtype, device=device)
                
                targets = [{
                    "boxes": t["boxes"] / scale,
                    "labels": t["labels"],
                } for t in targets
            ]
                class_pred, box_pred = model(images)

                indices = matcher(class_pred, box_pred, targets)

                loss_class = class_loss(class_pred, targets, indices, model.class_count)
                loss_bbox = bbox_loss(box_pred, targets, indices)
                loss_IoU = iou_loss(box_pred, targets, indices)
                loss = loss_class + loss_bbox + loss_IoU

                validate_loss+=loss.item()
                counter += 1

        validate_loss /= len(loaders["validate"])
        print("Validation Complete.")

        print(f"End Epoch {epoch + 1} of {num_epochs}")
        print(f"Training Loss: {training_loss:0.3f}\t Validate Loss: {validate_loss:0.3f}")
        epoch_data.append((round(training_loss,3), round(validate_loss,3)))


        torch.save(model.state_dict(),f"tmp_weights{str(epoch)}.pt")
        
        
        #check_annotation(0, training_dataset)
    