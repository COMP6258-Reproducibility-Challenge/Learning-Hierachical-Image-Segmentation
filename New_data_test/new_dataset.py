import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode, PILToTensor
import matplotlib.pyplot as plt  
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMAGE_DIR = 'Images_cell'      # replace with breast
MASK_DIR  = 'masks_cell'     
CHECKPOINT_PATH = 'model_weights/CAST-B/checkpoint_0099.pth.tar'
VIS_DIR         = ''     # replace with own output route

IMG_SIZE    = 224
BATCH_SIZE  = 8
NUM_WORKERS = 4
LR          = 1e-4
NUM_EPOCHS  = 15
NUM_CLASSES = 2 

class BreastCancerSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_transform, mask_transform):
        self.img_paths      = img_paths
        self.mask_paths     = mask_paths
        self.img_transform  = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.img_transform(img)
        msk = Image.open(self.mask_paths[idx]).convert('L')
        msk = self.mask_transform(msk)      
        msk = (msk > 0).long().squeeze(0)     
        return img, msk

class CastSegFormer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from cast_models.cast import cast_base
        self.backbone = cast_base(pretrained=False)
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = {
            k.replace('module.base_encoder.', ''): v
            for k, v in ckpt['state_dict'].items()
            if 'module.base_encoder.' in k and 'head' not in k
        }
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.head = nn.Identity()
        embed_dim = self.backbone.embed_dim
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.backbone.patch_embed(x)     
        feat = patches.permute(0, 3, 1, 2)           
        logits = self.seg_head(feat)                    
        return F.interpolate(logits, size=(H, W),  
                             mode='bilinear',
                             align_corners=False)

def dice_loss(logits, target, eps=1e-6):
    probs = F.softmax(logits, dim=1)[:,1]       
    target_f = target.float()
    inter = (probs * target_f).sum(dim=(1,2))
    union = probs.sum(dim=(1,2)) + target_f.sum(dim=(1,2))
    loss = 1 - ((2 * inter + eps) / (union + eps))
    return loss.mean()

def compute_global_iou(model, loader, device):
    inter_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    union_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for imgs, msks in loader:
            imgs, msks = imgs.to(device), msks.to(device)
            preds = model(imgs).argmax(dim=1)
            for cls in range(NUM_CLASSES):
                p = (preds == cls)
                t = (msks   == cls)
                inter_sum[cls] += (p & t).sum().item()
                union_sum[cls] += (p | t).sum().item()
    ious = inter_sum / (union_sum + 1e-6)
    return ious, np.nanmean(ious)

def main():
    img_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
    msk_paths = [os.path.join(MASK_DIR, os.path.basename(p)) for p in img_paths]
    data = list(zip(img_paths, msk_paths))
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data   = data[split:]
    train_imgs, train_msks = zip(*train_data)
    val_imgs,   val_msks   = zip(*val_data)

    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
        PILToTensor(),  # uint8 [1,H,W]
    ])

    train_ds = BreastCancerSegDataset(train_imgs, train_msks,
                                      img_transform, mask_transform)
    val_ds   = BreastCancerSegDataset(val_imgs,   val_msks,
                                      img_transform, mask_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model     = CastSegFormer(NUM_CLASSES).to(device)
    weights   = torch.tensor([1.0, 5.0], device=device)
    ce_loss   = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_miou = 0.0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss_ce   = ce_loss(logits, msks)
            loss_dice = dice_loss(logits, msks)
            loss      = loss_ce + loss_dice
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        running_loss /= len(train_loader.dataset)

        per_cls_iou, miou = compute_global_iou(model, val_loader, device)
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_cast_seg.pth')
        print(f"Epoch {epoch:02d} | Loss: {running_loss:.4f} | mIoU: {miou:.4f} | IoU per class: {per_cls_iou}")
    print(f"\n Best mIoU = {best_miou:.4f}")

    os.makedirs(VIS_DIR, exist_ok=True)
    indices = random.sample(range(len(val_imgs)), k=min(20, len(val_imgs)))
    for idx in indices:
        pil_img = Image.open(val_imgs[idx]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        inp     = img_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp).argmax(dim=1).squeeze(0).cpu().numpy()

        orig_np    = np.array(pil_img, dtype=np.uint8)
        mask_color = np.zeros_like(orig_np)
        mask_color[pred == 1] = [255, 0, 0]
        overlay = (orig_np * 0.5 + mask_color * 0.5).astype(np.uint8)

        base = os.path.splitext(os.path.basename(val_imgs[idx]))[0]
        out_path = os.path.join(VIS_DIR, f'vis_{base}.png')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_np);        axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(pred, cmap='gray'); axes[1].set_title('Pred Mask'); axes[1].axis('off')
        axes[2].imshow(overlay);        axes[2].set_title('Overlay');   axes[2].axis('off')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization: {out_path}")

if __name__ == '__main__':
    main()