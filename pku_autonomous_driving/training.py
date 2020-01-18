from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
from .const import SWITCH_LOSS_EPOCH
from .dataset import CarDataset
from .io import DataRecord
import gc
import pickle

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

use_apex = torch.cuda.is_available() and has_apex

def criterion(prediction, mask, regr, weight=0.4, size_average=True, lr=1.0):
    eps = 1e-7
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    #mask_loss = mask * torch.log(pred_mask + eps) + (1 - mask) * torch.log(
    #    1 - pred_mask + eps
    #)
    #mask_loss = -mask_loss.mean(0).sum()
    mask_loss = (mask == 1) * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (mask != 1) * (1 - mask) ** 4 *  pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum() / ((mask == 1).sum() + eps)

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    #print(pred_regr.shape, regr.shape)
    dxyz = torch.abs(pred_regr[:, :3] - regr[:, :3]).sum(1)
    dyaw_cos = torch.abs(pred_regr[:, 3] - regr[:, 3])
    dpitch_cos = torch.abs(torch.cos(pred_regr[:, 4]) - torch.cos(regr[:, 4]))
    dpitch_sin = torch.abs(torch.cos(pred_regr[:, 4]) - torch.cos(regr[:, 4]))
    droll = torch.abs(pred_regr[:, 5] - regr[:, 5])
    #regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / (mask.sum(1).sum(1) + eps)
    dsum = dxyz + dyaw + dpitch_sin + dpitch_cos + droll
    #print(dxyz.sum(), dyaw.sum() ,dpitch_cos.sum(), dpitch_sin.sum(), droll.sum())

    regr_loss = (dsum * mask).sum(1).sum(1) / (mask.sum(1).sum(1) + eps)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = mask_loss + lr * regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss


def clean_up():
    torch.cuda.empty_cache()
    gc.collect()


def train(model, optimizer, scheduler, train_loader, epoch, device, history=None, lr=1.0):
    model.train()
    t = tqdm(train_loader)
    for batch_idx, input in enumerate(t):
        img_batch = input["img"].to(device)
        mask_batch = input["mask"].to(device)
        regr_batch = input["regr"].to(device)
        torch.cuda.empty_cache()

        output = model(img_batch)
        weight = 1.0 if epoch < SWITCH_LOSS_EPOCH else 0.5
        loss, mask_loss, regr_loss = criterion(
            output, mask_batch, regr_batch, weight=weight, lr=lr
        )

        t.set_description(
            f"train_loss (l={loss:.3f})(m={mask_loss:.2f}) (r={regr_loss:.4f}"
        )

        if history is not None:
            history.loc[
                epoch + batch_idx / len(train_loader), "train_loss"
            ] = loss.data.cpu().numpy()

        optimizer.zero_grad()
        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

    print(
        "Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}\tMaskLoss: {:.6f}\tRegLoss: {:.6f}".format(
            epoch,
            optimizer.state_dict()["param_groups"][0]["lr"],
            loss.data,
            mask_loss.data,
            regr_loss.data,
        )
    )


def evaluate(model, dev_loader, epoch, device, history=None, lr=1.0):
    model.eval()
    loss = 0
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    with torch.no_grad():
        for input in dev_loader:
            img_batch = input["img"].to(device)
            mask_batch = input["mask"].to(device)
            regr_batch = input["regr"].to(device)

            output = model(img_batch)

            weight = 1.0 if epoch < SWITCH_LOSS_EPOCH else 0.5
            loss, mask_loss, regr_loss = criterion(
                output, mask_batch, regr_batch, weight=weight, size_average=False, lr=lr
            )
            valid_loss += loss.data
            valid_mask_loss += mask_loss.data
            valid_regr_loss += regr_loss.data

    valid_loss /= len(dev_loader.dataset)
    valid_mask_loss /= len(dev_loader.dataset)
    valid_regr_loss /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch, "dev_loss"] = valid_loss.cpu().numpy()
        history.loc[epoch, "mask_loss"] = valid_mask_loss.cpu().numpy()
        history.loc[epoch, "regr_loss"] = valid_regr_loss.cpu().numpy()

    print("Dev loss: {:.4f}".format(valid_loss))


def save_checkpoint(model, optimizer, history):
    with open('./history.pickle', 'wb') as fp:
        pickle.dump(history , fp)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if use_apex:
        checkpoint['amp'] = amp.state_dict()
    torch.save(checkpoint, f'checkpoint.pt')


def setup(model: nn.Module, optimizer, device, path: Path = None, opt_level='O1'):
    checkpoint = {}
    if path is not None:
        checkpoint = torch.load(str(path))

    if use_apex:
        model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'amp' in checkpoint:
        amp.load_state_dict(checkpoint['amp'])

    if not use_apex:
        model.to(device)
    return model, optimizer
