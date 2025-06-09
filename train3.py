import os
import argparse
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from datasets.cityscapes.dataloader.get_dataloaders import return_dataloader
from models.ffnet_gpu_small import segmentation_ffnet18_dAAC
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


NUM_CLASSES, IGNORE_INDEX = 3, 255

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--output_dir", type=str, default="weight")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.makedirs(args.output_dir, exist_ok=True)

model = segmentation_ffnet18_dAAC().to(device)
dataloader = return_dataloader(
    batch_size=args.batch_size, num_workers=args.num_workers, mode="train"
)
val_loader = return_dataloader(
    batch_size=args.batch_size, num_workers=args.num_workers, mode="val"
)
loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=4e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scaler = GradScaler()
writer = SummaryWriter(args.output_dir)

best_miou = 0

print(model)


def get_miou(cm):
    intersection = torch.diag(cm)
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
    iou = intersection / (union.float() + 1e-6)
    return torch.mean(iou[union > 0]).item() * 100


def eval(model, dataloader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), device=device, dtype=torch.long
    )
    with torch.no_grad():
        for images, labels, _, _, _ in dataloader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            logits = F.interpolate(
                model(images),
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            total_loss += loss_fn(logits, labels).item()

            mask = labels != IGNORE_INDEX
            preds = torch.argmax(logits, dim=1)
            confusion_matrix += torch.bincount(
                (labels[mask] * num_classes + preds[mask]), minlength=num_classes**2
            ).reshape(num_classes, num_classes)

    return total_loss / len(dataloader), get_miou(confusion_matrix)


for epoch in range(150):
    model.train()
    epoch_loss = 0.0
    confusion_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), device=device, dtype=torch.long
    )

    for i, (images, labels, _, _, _) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = F.interpolate(
                model(images),
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        mask = labels != IGNORE_INDEX
        preds = torch.argmax(logits.detach(), dim=1)
        confusion_matrix += torch.bincount(
            (labels[mask] * NUM_CLASSES + preds[mask]), minlength=NUM_CLASSES**2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + i)
        writer.add_scalar(
            "mIoU/train", get_miou(confusion_matrix), epoch * len(dataloader) + i
        )
        print(
            f"E:{epoch+1}, I:{i}/{len(dataloader)} | Loss:{loss.item():.4f} | mIoU:{get_miou(confusion_matrix):.2f}%"
        )

    val_loss, val_miou = eval(model, val_loader, loss_fn, device, NUM_CLASSES)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("mIoU/val", val_miou, epoch)
    print(
        f"--- Epoch {epoch+1} Summary: Val Loss:{val_loss:.4f} | Val mIoU:{val_miou:.2f}% ---"
    )

    if val_miou > best_miou:
        best_miou = val_miou
        save_path = os.path.join(args.output_dir, f"model_{int(val_miou)}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"*** New best saved to {save_path} with mIoU: {val_miou:.2f}% ***")
    scheduler.step()
    writer.flush()
writer.close()
