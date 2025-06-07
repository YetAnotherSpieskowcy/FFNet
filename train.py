import logging
import argparse

import torch
from rmi import RMILoss

from datasets.cityscapes.dataloader.get_dataloaders import return_dataloader
from datasets.cityscapes.utils.trnval_utils import resize_tensor
from models.ffnet_S_gpu_small import segmentation_ffnet150S_dBBB


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Training of FFNet'
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the input")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    args = parser.parse_args()


    logger = logging.getLogger("FFNet")
    logging.basicConfig(filename='../ffnet_weights/train.log', level=logging.DEBUG)

    model = segmentation_ffnet150S_dBBB()
    model.to("cuda")
    dataloader = return_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, mode="train")

    loss_function = RMILoss(with_logits=True)
    optim = torch.optim.RMSprop(model.parameters(), lr=1e-6, eps=1e-3, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=2)

    for epoch in range(150):
        it = 0
        for batch, (images, labels, _,  _, _) in enumerate(dataloader):
            it+=1
            images = images.to("cuda")
            labels = labels.to("cuda")
            pred_labels = model(images)

            input_size = images.size(2), images.size(3)
            output = resize_tensor(pred_labels.long(), input_size, True)
            output_data = torch.nn.functional.softmax(output, dim=1).data
            max_probs, predictions = output_data.max(1)

            predictions = predictions.unsqueeze(0)
            labels = labels.unsqueeze(0)

            optim.zero_grad()
            loss = loss_function(predictions, labels)
            optim.step()

            mes = f"Epoch {epoch+1}, iter {it}: loss {loss}"
            print(mes)
            logger.debug(mes)

        scheduler.step()

    torch.save(model, "../ffnet_weights/model.pth")
