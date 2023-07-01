# coding=utf-8


from __future__ import absolute_import, division, print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import HParam
from model.cnn_transformer import CNN_Transformer
from utils.utils import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.utils import get_world_size, get_rank


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()




def valid(device, local_rank, hp, model, writer, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
   
    

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    print('accuracy',accuracy)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    

    return accuracy







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.") 
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    hp = HParam(args.config)
    device = torch.device('cuda:{:d}'.format(0))
    model = CNN_Transformer(image_size = hp.data.image_size, patch_size = hp.model.patch_size, num_classes =      hp.model.num_classes, 
                 dim = hp.model.dim, depth = hp.model.depth, heads = hp.model.heads, pool = hp.model.pool, 
                 in_channels = hp.model.in_channels, out_channels = hp.model.out_channels, with_lca=hp.model.with_lca)
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    
    train_loader, test_loader = get_loader(0, hp)
    
    accuracy = valid(device, 0, hp, model, writer, test_loader)
    
