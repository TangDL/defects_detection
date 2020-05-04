import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder720p
from utils import get_config, get_args, dump_cfg
from utils import save_imgs

# models
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))

from cae_classify import CAE_Classify


def prologue(cfg: Namespace, *varargs) -> SummaryWriter:
    # sanity checks
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    # dirs
    base_dir = f"/data2/TDL/paper_fabric/workdir/4_30_short_eassy/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)
    os.makedirs(f"{base_dir}/chkpt", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)

    dump_cfg(f"{base_dir}/train_config.txt", vars(cfg))

    # tb writer
    writer = SummaryWriter(f"{base_dir}/logs")

    return writer


def epilogue(cfg: Namespace, *varargs) -> None:
    writer = varargs[0]
    writer.close()


def train(cfg: Namespace) -> None:
    logger.info("=== Training ===")

    # initial setup
    writer = prologue(cfg)

    # train-related code
    model = CAE_Classify()
    model.load_state_dict(torch.load(cfg.load_from), strict=False)     # 加载预训练模型
    model.train()
    if cfg.device == "cuda":
        model.cuda()
    logger.debug(f"Model loaded on {cfg.device}")

    dataset = ImageFolder720p(cfg.dataset_path)
    test_dataset = ImageFolder720p(cfg.test_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    logger.debug("Data loaded")

    for index, data in enumerate(test_dataloader):
        if index > 0:
            break
        _, test_y, test_pathch, _ = data



    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.learning_rate, weight_decay=1e-5)
    loss_criterion = torch.nn.CrossEntropyLoss()
    # scheduler = ...

    avg_loss, epoch_avg = 0.0, 0.0
    ts = 0

    # train-loop
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):

        # scheduler.step()

        for batch_idx, data in enumerate(dataloader, start=1):
            img, y, patches, _ = data

            if cfg.device == "cuda":
                patches = patches.cuda()

            avg_loss_per_image = 0.0        # 初始化单张图片的损失
            optimizer.zero_grad()

            x = Variable(patches[:, :, 0, 0, :, :])
            pred_y = model(x)

            # loss = F.nll_loss(pred_y, y.cuda())
            loss = loss_criterion(pred_y, y.cuda())

            avg_loss_per_image += loss.item()

            loss.backward()
            optimizer.step()

            avg_loss += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.batch_every == 0:
                writer.add_scalar("train/avg_loss", avg_loss / cfg.batch_every, ts)

                x = Variable(test_pathch[:, :, 0, 0, :, :]).cuda()
                test_output = model(x).cpu()
                pred_y = torch.max(test_output, 1)[1].cpu().numpy()
                accuracy = float((pred_y == test_y.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))


                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, ts)

                logger.debug(
                    '[%3d/%3d][%5d/%5d] avg_loss: %.8f accuracy: %.8f'%
                    (epoch_idx, cfg.num_epochs, batch_idx, len(dataloader), avg_loss / cfg.batch_every, accuracy)
                )
                avg_loss = 0.0
                ts += 1


        # -- batch-loop

        if epoch_idx % cfg.epoch_every == 0:
            epoch_avg /= (len(dataloader) * cfg.epoch_every)
            accuracy = 0
            for index, data in enumerate(test_dataloader):
                _, test_y, patches, _ = data
                x = Variable(patches[:, :, 0, 0, :, :]).cuda()
                test_output = model(x)
                pred_y = torch.max(test_output, 1)[1]
                accuracy += float((test_y.cpu().numpy() == pred_y).astype().sum()) / float(test_y.size(0))
            accuracy /= (index - 1)

            writer.add_scalar("train/epoch_avg_loss", avg_loss / cfg.batch_every, epoch_idx // cfg.epoch_every)

            logger.info("Epoch avg = %.8f  Accuracy = %.8f" % epoch_avg, accuracy)
            epoch_avg = 0.0
            torch.save(model.state_dict(), f"/data2/TDL/paper_fabric/workdir/4_30_short_eassy/{cfg.exp_name}/chkpt/model_{epoch_idx}.pth")

    # -- train-loop

    # save final model
    torch.save(model.state_dict(), f"/data2/TDL/paper_fabric/workdir/4_30_short_eassy/{cfg.exp_name}/model_final.pth")

    # final setup
    epilogue(cfg, writer)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args)

    train(config)
