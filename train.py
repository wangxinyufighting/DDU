"""
Script for training a single model for OOD detection.
"""

import json
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.ood_detection.coco as coco
import data.ood_detection.pascal_voc as pascal
import data.dirty_mnist as dirty_mnist
import data.datsest_multi_label as datsest_multi_label


# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import model_save_name
from utils.train_utils import train_single_epoch, test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter


dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10, "coco": 80, 'pascal': 20}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,
    "coco": coco,
    "pascal": pascal
}

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": .gitignoreresnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}


if __name__ == "__main__":

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=num_classes,
        mnist="mnist" in args.dataset,
    )

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    opt_params = net.parameters()
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1
    )

    if args.dataset in ['pascal']:
        P = {}
        P['dataset'] = args.dataset
        P['val_frac'] = 0.2
        P['split_seed'] = 1200
        P['train_set_variant'] = 'observed'
        P['ss_frac_train'] = 1.0
        P['val_set_variant'] = 'clean'
        P['use_feats'] = False
        P['ss_seed'] = 999
        P['ss_frac_val'] = 1.0
        P['bsize'] = args.train_batch_size
        P['num_workers'] = 4 
        dataset = datsest_multi_label.get_data(P)
        dataloader = {}

        train_loader = torch.utils.data.DataLoader(
                dataset['train'],
                batch_size = P['bsize'],
                shuffle = 'train' == 'train',
                sampler = None,
                num_workers = P['num_workers'],
                drop_last = False,
                pin_memory = True
            )
    else:
        train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
            root=args.dataset_root,
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            val_size=0.1,
            val_seed=args.seed,
            pin_memory=args.gpu,
        )

    # Creating summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")

    training_set_loss = {}

    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
    print("Model save name", save_name)

    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        train_loss = train_single_epoch(
            epoch, net, train_loader, optimizer, device, loss_function=args.loss_function, loss_mean=args.loss_mean,
        )

        training_set_loss[epoch] = train_loss
        writer.add_scalar(save_name + "_train_loss", train_loss, (epoch + 1))

        scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
            torch.save(net.state_dict(), saved_name)

    saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)

    writer.close()
    with open(saved_name[: saved_name.rfind("_")] + "_train_loss.json", "a") as f:
        json.dump(training_set_loss, f)
