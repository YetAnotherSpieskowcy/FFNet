import os
import argparse
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from datasets.cityscapes.dataloader.get_dataloaders import return_dataloader
from models.ffnet_S_gpu_small import segmentation_ffnet150S_dBBB

NUM_CLASSES, IGNORE_INDEX = 19, 255

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--output_dir", type=str, default="weight")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.makedirs(args.output_dir, exist_ok=True)

model = segmentation_ffnet150S_dBBB().to(device)
dataloader = return_dataloader(
    batch_size=args.batch_size, num_workers=args.num_workers, mode="train"
)
loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=4e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

last_miou = 0


def get_miou(cm):
    intersection = torch.diag(cm)
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
    iou = intersection / (union.float() + 1e-6)
    return torch.mean(iou[union > 0]).item() * 100


for epoch in range(150):
    model.train()
    epoch_loss = 0.0
    confusion_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), device=device, dtype=torch.long
    )

    for i, (images, labels, _, _, _) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = F.interpolate(
            model(images),
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        mask = labels != IGNORE_INDEX
        preds = torch.argmax(logits.detach(), dim=1)
        confusion_matrix += torch.bincount(
            (labels[mask] * NUM_CLASSES + preds[mask]), minlength=NUM_CLASSES**2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        if i % 50 == 0:
            print(
                f"E:{epoch+1}, I:{i}/{len(dataloader)} | Loss:{loss.item():.4f} | mIoU:{get_miou(confusion_matrix):.2f}%"
            )

    print(
        f"--- E {epoch+1}: Avg Loss:{(epoch_loss/len(dataloader)):.4f} | mIoU:{get_miou(confusion_matrix):.2f}% ---"
    )
    last_miou = get_miou(confusion_matrix)
    scheduler.step()

save_path = os.path.join(args.output_dir, f"model_{int(last_miou)}.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
