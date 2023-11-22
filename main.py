import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from loss.contrastive import BalSCL, SCL
from loss.logitadjust import LogitAdjust, FocalLoss, FocalLC, EQLv2, LabelSmoothingCrossEntropy
import math
from tensorboardX import SummaryWriter
from dataset.mydataset import MyDataset
from models import resnext
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision
from utils import GaussianBlur, shot_acc, get_random_string
import argparse
import random
import string
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
import time
import wandb
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="isic", choices=["inat", "isic", "aptos"])
parser.add_argument(
    "--data",
    default="/l/users/salwa.khatib/proco/ISIC2018_Task3_Training_Input/",
    metavar="DIR",
)
parser.add_argument(
    "--val_data",
    default="/l/users/salwa.khatib/proco/ISIC2018_Task3_Validation_Input/",
    metavar="DIR",
)
parser.add_argument(
    "--val_txt",
    default="/l/users/salwa.khatib/proco/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_GroundTruth.txt",
    metavar="Path",
)
parser.add_argument(
    "--txt",
    default="/l/users/salwa.khatib/proco/ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_GroundTruth.txt",
    metavar="Path",
)
parser.add_argument(
    "--arch",
    default="resnext50",
    choices=["resnet50", "resnext50", "crossformer", "vit_small"],
)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--epochs", default=90, type=int)
parser.add_argument("--classes", default=7, type=int)
parser.add_argument(
    "--temp",
    default=0.07,
    type=float,
    help="scalar temperature for contrastive learning",
)
parser.add_argument(
    "--delayed_start",
    default=False,
    type=bool,
    help="start contrastive learning delayed",
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[860, 880],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print_freq",
    default=3,
    type=int,
    metavar="N",
    help="print frequency (default: 20)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--alpha", default=1.0, type=float, help="cross entropy loss weight"
)
parser.add_argument(
    "--beta", default=0.35, type=float, help="supervised contrastive loss weight"
)
parser.add_argument(
    "--randaug",
    default=True,
    type=bool,
    help="use RandAugmentation for classification branch",
)
parser.add_argument(
    "--cl_views",
    default="sim-sim",
    type=str,
    choices=["sim-sim", "sim-rand", "rand-rand"],
    help="Augmentation strategy for contrastive learning views",
)
parser.add_argument(
    "--feat_dim", default=1024, type=int, help="feature dimension of mlp head"
)
parser.add_argument("--warmup_epochs", default=0, type=int, help="warmup epochs")
parser.add_argument("--root_log", type=str, default="log")
parser.add_argument(
    "--cos", default=True, type=bool, help="lr decays by cosine scheduler. "
)
parser.add_argument("--use_norm", default=True, type=bool, help="cosine classifier.")
parser.add_argument("--randaug_m", default=10, type=int, help="randaug-m")
parser.add_argument("--randaug_n", default=2, type=int, help="randaug-n")
parser.add_argument(
    "--many_shot_thr", default=1000, type=int, help="many shot threshold"
)
parser.add_argument("--low_shot_thr", default=200, type=int, help="low shot threshold")
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training"
)
parser.add_argument("--reload", default=False, type=bool, help="load supervised model")
parser.add_argument(
    "--recalibrate", default=False, type=bool, help="recalibrate prototypes"
)
parser.add_argument(
    "--recalibrate_static",
    default=False,
    type=bool,
    help="recalibrate prototypes statically"
)
parser.add_argument(
    "--ce_loss",
    default="LC",
    choices=["LC", "Focal", "FocalLC", "LS", "EQLv2", "WeightedBCE"],
    help="type of cross entropy loss",
)
parser.add_argument(
    "--loss_req",
    default="BCL",
    choices=["BCL", "Supcon", "LC"],
    help="type of loss requirement",
)
parser.add_argument(
    "--logit_adjust",
    default="train",
    choices=["train", "val"],
    help="do logit compensation based on which set",
)
parser.add_argument(
    "--user_name",
    default="mai",
    type=str,
    help="wandb user name",
    # choices=["salwa", "mai", "dana"],
)
parser.add_argument(
    "--pretrained", default=False, type=bool, help="use pretrained model"
)
parser.add_argument(
    "--ema_prototypes", default=False, type=bool, help="use ema updates to get prototypes"
)

def main():
    """
    This function initializes and logs hyperparameters and runs training and logs using the Weights and Biases
    (wandb) library.
    """
    args = parser.parse_args()
    user_name = args.user_name
    wandb_entity = "none"

    if user_name == "salwa":
        wandb_entity = "salwa-khatib"
    elif user_name == "mai":
        wandb_entity = "mai-cs"
    elif user_name == "dana":
        wandb_entity = "danaosama"
    else:
        wandb_entity = user_name

    wandb.login()
    args.store_name = "_".join(
        [
            args.dataset,
            args.arch,
            "batchsize",
            str(args.batch_size),
            "epochs",
            str(args.epochs),
            "temp",
            str(args.temp),
            "lr",
            str(args.lr),
            args.cl_views,
            "alpha",
            str(args.alpha),
            "beta",
            str(args.beta),
            "schedule",
            str(args.schedule),
            "recalibrate-beta0.99",
            str(args.recalibrate),
            user_name,
            "ce_loss",
            str(args.ce_loss),
            "pretrained",
            str(args.pretrained),
            "prototype_ema",
            str(args.ema_prototypes),
            get_random_string(6),
        ]
    )
    print("storing name: {}".format(args.store_name))
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.dataset,
        # track hyperparameters and run metadata
        config={
            "architecture": args.arch,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "start_epoch": args.start_epoch,
            "max_epochs": args.epochs,
            "learning_rate": args.lr,
            "delayed_start": args.delayed_start,
            "momentum": args.momentum,
            "schedule": args.schedule,
            "weight_decay": args.weight_decay,
            "temp": args.temp,
            "alpha": args.alpha,  # cross entropy loss weight
            "beta": args.beta,  # supervised contrastive loss weight
            "cl_views": args.cl_views,  # Augmentation strategy for contrastive learning views
            "seed": args.seed,
            "randaug": args.randaug,
            "randaug_m": args.randaug_m,
            "randaug_n": args.randaug_n,
            "cos": args.cos,  # lr decays by cosine scheduler.
            "use_norm": args.use_norm,
            "feat_dim": args.feat_dim,  # feature dimension of mlp head
            "warmup_epochs": args.warmup_epochs,
            "recalibrate": args.recalibrate,
            "ce_loss": args.ce_loss,
            "logit_adjust": args.logit_adjust,
            "many_shot_thr": args.many_shot_thr,
            "low_shot_thr": args.low_shot_thr,
            "resume": args.resume,
            "gpu": args.gpu,
            "pretrained": args.pretrained,
        },
        # entity='bcl',
        entity=wandb_entity,
        name=args.store_name,
    )

    print("[INFO] Wandb initialized!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
    ngpus_per_node = torch.cuda.device_count()

    start_time = time.time()
    main_worker(args.gpu, ngpus_per_node, args)
    print("total time: {:.2f}".format((time.time() - start_time) / 60))

def main_worker(gpu, ngpus_per_node, args):
    """
    This function defines the main worker for a PyTorch training script, including setting up data
    loading, model creation, loss functions, and training/validation loops.
    
    :param gpu: The GPU device ID to use for training. If None, all available GPUs will be used
    :param ngpus_per_node: `ngpus_per_node` is the number of GPUs available per node in the distributed
    training setup
    :param args: `args` is an object that contains various arguments and hyperparameters for the
    training process. These include the dataset being used, the architecture of the model, the learning
    rate, the number of epochs, the batch size, the number of workers for data loading, and various
    options for loss functions and data
    :return: The function `main_worker` does not have a return statement, so it does not return
    anything.
    """

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    ##########################################
    # The Salwa Branch
    if (args.user_name).lower() == "salwa":
        if args.dataset == "isic":
            args.data = "/l/users/salwa.khatib/proco/ISIC2018_Task3_Training_Input"
            args.val_data = (
                "/l/users/salwa.khatib/proco/ISIC2018_Task3_Validation_Input"
            )
            txt_train = f"/l/users/salwa.khatib/proco/ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_GroundTruth.txt"
            txt_val = f"/l/users/salwa.khatib/proco/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_GroundTruth.txt"
        elif args.dataset == "aptos":
            args.data = "/l/users/salwa.khatib/aptos/train_images"
            args.val_data = "/l/users/salwa.khatib/aptos/train_images"
            txt_train = f"/l/users/salwa.khatib/aptos/train.txt"
            txt_val = f"/l/users/salwa.khatib/aptos/val.txt"
        else:
            txt_train = f"dataset/iNaturalist18/iNaturalist18_train.txt"
            txt_val = f"dataset/iNaturalist18/iNaturalist18_val.txt"

    # The Dana Branch
    elif (args.user_name).lower() == "dana":
        if args.dataset == "isic":
            args.data = "/nfs/users/ext_group6/data/ISIC2018_Task3_Training_Input"
            args.val_data = "/nfs/users/ext_group6/data/ISIC2018_Task3_Validation_Input"
            txt_train = (
                f"/nfs/users/ext_group6/data/ISIC2018_Task3_Training_GroundTruth.txt"
            )
            txt_val = (
                f"/nfs/users/ext_group6/data/ISIC2018_Task3_Validation_GroundTruth.txt"
            )
        elif args.dataset == "aptos":
            args.data = (
                "/nfs/users/ext_group6/data/aptos2019-blindness-detection/train_images"
            )
            args.val_data = (
                "/nfs/users/ext_group6/data/aptos2019-blindness-detection/train_images"
            )
            txt_train = f"/nfs/users/ext_group6/data/aptos2019-blindness-detection/aptos-split/train.txt"
            txt_val = f"/nfs/users/ext_group6/data/aptos2019-blindness-detection/aptos-split/val.txt"
        else:
            txt_train = f"dataset/iNaturalist18/iNaturalist18_train.txt"
            txt_val = f"dataset/iNaturalist18/iNaturalist18_val.txt"

    # The Mai Branch
    elif (args.user_name).lower() == "mai":
        if args.dataset == "isic":
            args.data = "/nfs/users/ext_group6/data/ISIC2018_Task3_Training_Input"
            args.val_data = "/nfs/users/ext_group6/data/ISIC2018_Task3_Validation_Input"
            txt_train = (
                f"/nfs/users/ext_group6/data/ISIC2018_Task3_Training_GroundTruth.txt"
            )
            txt_val = (
                f"/nfs/users/ext_group6/data/ISIC2018_Task3_Validation_GroundTruth.txt"
            )
        elif args.dataset == "aptos":
            args.data = "/l/users/salwa.khatib/aptos/train_images"
            args.val_data = "/l/users/salwa.khatib/aptos/train_images"
            txt_train = f"/l/users/salwa.khatib/aptos/train.txt"
            txt_val = f"/l/users/salwa.khatib/aptos/val.txt"
        else:
            txt_train = f"dataset/iNaturalist18/iNaturalist18_train.txt"
            txt_val = f"dataset/iNaturalist18/iNaturalist18_val.txt"
    
    else:
        txt_train = args.txt
        txt_val = args.val_txt

    # Transformations and data loaders
    if args.dataset == "isic":
        normalize = transforms.Normalize(
            (0.7635, 0.5461, 0.5705), (0.0896, 0.1183, 0.1329)
        )
    elif args.dataset == "aptos":
        normalize = transforms.Normalize(
            (0.4138, 0.2210, 0.0737), (0.2389, 0.1320, 0.0497)
        )
    elif args.dataset == "inat":
        normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
    else:
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    rgb_mean = (0.485, 0.456, 0.406)

    ra_params = dict(
        translate_const=int(224 * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )

    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
        rand_augment_transform(
            "rand-n{}-m{}-mstd0.5".format(args.randaug_n, args.randaug_m), ra_params
        ),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        rand_augment_transform(
            "rand-n{}-m{}-mstd0.5".format(args.randaug_n, args.randaug_m), ra_params
        ),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ]

    if args.cl_views == "sim-sim":
        transform_train = [
            transforms.Compose(augmentation_randncls),
            transforms.Compose(augmentation_sim),
            transforms.Compose(augmentation_sim),
        ]
    elif args.cl_views == "sim-rand":
        transform_train = [
            transforms.Compose(augmentation_randncls),
            transforms.Compose(augmentation_randnclsstack),
            transforms.Compose(augmentation_sim),
        ]
    elif args.cl_views == "randstack-randstack":
        transform_train = [
            transforms.Compose(augmentation_randncls),
            transforms.Compose(augmentation_randnclsstack),
            transforms.Compose(augmentation_randnclsstack),
        ]

    else:
        raise NotImplementedError(
            "This augmentations strategy is not available for contrastive learning branch!"
        )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_dataset = MyDataset(
        root=args.val_data,
        txt=txt_val,
        transform=val_transform,
        train=False,
        num_classes=args.classes,
    )
    train_dataset = MyDataset(
        root=args.data,
        txt=txt_train,
        transform=transform_train,
        num_classes=args.classes,
    )

    if args.logit_adjust == "train":
        cls_num_list = train_dataset.cls_num_list
    elif args.logit_adjust == "val":
        cls_num_list = val_dataset.cls_num_list

    args.cls_num = len(cls_num_list)
    print("[INFO] cls_num_list:", cls_num_list)
    print("[INFO] cls_num:", args.cls_num)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Create model
    print("=> creating model '{}'".format(args.arch))
    if(args.ema_prototypes):
        print("=> using EMA prototypes")
    if args.arch == "resnet50":
        model = resnext.BCLModel(
            name="resnet50",
            num_classes=args.cls_num,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
            recalibrate=args.recalibrate,
            pretrained=args.pretrained,
            ema_prototypes=args.ema_prototypes,
            cls_num_list=cls_num_list,
            static = args.recalibrate_static
        )
    elif args.arch == "resnext50":
        model = resnext.BCLModel(
            name="resnext50",
            num_classes=args.cls_num,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
            recalibrate=args.recalibrate,
            pretrained=args.pretrained,
            ema_prototypes=args.ema_prototypes,
            cls_num_list=cls_num_list,
            static = args.recalibrate_static
        )
    elif args.arch == "crossformer":
        model = resnext.BCLModel(
            name="crossformer",
            num_classes=args.cls_num,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
            recalibrate=args.recalibrate,
            pretrained=args.pretrained,
            ema_prototypes=args.ema_prototypes,
            cls_num_list=cls_num_list,
            static = args.recalibrate_static
        )
    elif args.arch == "vit_small":
        model = resnext.BCLModel(
            name="vit_small",
            num_classes=args.cls_num,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
            recalibrate=args.recalibrate,
            pretrained=args.pretrained,
            ema_prototypes=args.ema_prototypes,
            cls_num_list=cls_num_list,
            static = args.recalibrate_static
        )
    else:
        raise NotImplementedError("[ERROR] This model is not supported")
    print(model)
    print("[INFO] number of parameters: ", sum(p.numel() for p in model.parameters()))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("=> using all available GPUs")
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cuda:0")
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            wandb.log({"best_val_top1": best_acc1})
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
                print("best acc1 of resumed model: ", best_acc1)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise Exception("No checkpoint found")

    cudnn.benchmark = True

    # Initializing loss classes
    if args.ce_loss == "LC":
        print("[INFO] Using Logit Compensation with cross entropy loss")
        criterion_ce = LogitAdjust(cls_num_list).cuda(args.gpu)
    elif args.ce_loss == "Focal":
        print("[INFO] Using focal loss")
        criterion_ce = FocalLoss().cuda(args.gpu)
    elif args.ce_loss == "FocalLC":
        print("[INFO] Using focal loss with logit compensation")
        criterion_ce = FocalLC(cls_num_list).cuda(args.gpu)
    elif args.ce_loss == "LS":
        print("[INFO] Using label smoothing")
        criterion_ce = LabelSmoothingCrossEntropy().cuda(args.gpu)
    elif args.ce_loss == "EQLv2":
        print("[INFO] Using EQLv2 loss")
        criterion_ce = EQLv2(num_classes = args.classes).cuda(args.gpu)
    elif args.ce_loss == "WeightedBCE":
        print("[INFO] Using weighted BCE loss")
        class_weights = torch.tensor([1 - (x / sum(cls_num_list)) for x in cls_num_list], device=args.gpu)
        criterion_ce = nn.BCEWithLogitsLoss(pos_weight = class_weights).cuda(args.gpu)
    else:
        raise ValueError(f"{str(args.ce_loss)} not supported")

    # Balanced Contrastive loss
    criterion_scl = BalSCL(cls_num_list, args.temp).cuda(args.gpu)
    # criterion_scl = SCL().cuda(args.gpu)

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    if not args.resume:
        best_acc1 = 0.0
    best_many, best_med, best_few, best_f1 = 0.0, 0.0, 0.0, 0.0

    if args.reload:
        if args.dataset == "isic":
            txt_test = (
                f"/nfs/users/ext_group6/data/ISIC2018_Task3_Validation_GroundTruth.txt"
            )
        elif args.dataset == "aptos":
            txt_test = f"/l/users/salwa.khatib/aptos/val.txt"
        else:
            txt_test = f"dataset/iNaturalist18/iNaturalist18_val.txt"
        test_dataset = MyDataset(
            root=args.data, txt=txt_test, transform=val_transform, train=False
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        acc1, many, med, few = validate(
            train_loader, val_loader, model, criterion_ce, 1, args, tf_writer
        )
        print(
            "Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}".format(
                acc1, many, med, few
            )
        )
        return

    # we only watch gradients for now. To watch hyperparameters, use 'all'
    # log_freq: log gradients and parameters every N batches
    wandb.watch(model, log="gradients", log_freq=5)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)

        # train for one epoch
        targets, tsne_f1, tsne_f2 = train(
            train_loader,
            model,
            criterion_ce,
            criterion_scl,
            optimizer,
            epoch,
            args,
            tf_writer,
        )

        # evaluate on validation set
        acc1, f1, many, med, few, val_tsne_targets, val_tsne_f1 = validate(
            train_loader, val_loader, model, criterion_ce, epoch, args, tf_writer
        )

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
            best_f1 = f1

            # print when it updates
            print(
                "Best Prec@1: {:.3f}, Corresponding: F1 Score: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}".format(
                    best_acc1, best_f1, best_many, best_med, best_few
                )
            )
            wandb.log(
                {
                    "best_val_top1": best_acc1,
                    "best_epoch": epoch,
                    "corresponding_val_f1": best_f1,
                    "corresponding_many_val_top1": best_many,
                    "corresponding_med_val_top1": best_med,
                    "corresponding_few_val_top1": best_few,
                }
            )
        # save the last checkpoint, and if best, save as best
        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

        # report training time
        end_time = time.time()
    wandb.save("model.onnx")
    wandb.finish(exit_code=-1)

def train(
    train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args, tf_writer
):
    """
    This function trains a given model using a specified data loader, criterion, optimizer, and logging
    tools, and returns the targets and features for t-SNE visualization.
    
    :param train_loader: The data loader for training dataset, which provides batches of input data and
    their corresponding labels for the model to train on
    :param model: The neural network model being trained
    :param criterion_ce: The cross-entropy loss criterion used for training the model
    :param criterion_scl: The criterion used to calculate the SCL (Soft Center Loss) loss during
    training
    :param optimizer: The optimizer is an object that specifies the optimization algorithm used to
    update the model's parameters during training. It takes in the model's parameters and updates them
    based on the gradients computed during backpropagation. Examples of optimizers include stochastic
    gradient descent (SGD), Adam, and Adagrad
    :param epoch: The current epoch number of the training process
    :param args: args is a namespace object that contains various arguments passed to the function.
    These arguments are used to configure the training process, such as the number of classes, the type
    of loss function to use, the learning rate, etc
    :param tf_writer: tf_writer is a TensorBoard writer object used to write training logs and metrics
    to TensorBoard. It is used to visualize and track the training progress of the model
    :return: three tensors: tsne_targets, tsne_f1, and tsne_f2. These tensors are used for visualization
    purposes and are not directly related to the training process or the optimization of the model.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    ce_loss_all = AverageMeter("CE_Loss", ":.4e")
    scl_loss_all = AverageMeter("SCL_Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    f1 = AverageMeter("F1", ":.4e")

    model.train()
    end = time.time()

    tsne_f1 = []
    tsne_f2 = []
    tsne_targets = []

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = torch.cat([inputs[0], inputs[1], inputs[2]], dim=0)
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = targets.shape[0]
        feat_mlp, logits, centers = model(inputs, targets=targets, phase="train")

        centers = centers[: args.cls_num]
        f_1, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)

        tsne_f1.append(f_1)
        tsne_f2.append(f2)
        tsne_targets.append(targets)

        features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
        
        scl_loss = criterion_scl(centers, features, targets)
        # scl_loss = criterion_scl(features, targets)
        if(args.ce_loss == "WeightedBCE"):
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes=args.cls_num).float()
            ce_loss = criterion_ce(logits, targets_onehot)
        else:    
            ce_loss = criterion_ce(logits, targets)
        if(args.delayed_start and epoch > 50):
            loss = args.alpha * ce_loss  
        else:
            loss = args.alpha * ce_loss + args.beta * scl_loss

        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)
        acc1 = accuracy(logits, targets, topk=(1,))
        f1_acc = calc_f1(logits, targets)
        top1.update(acc1[0].item(), batch_size)
        f1.update(f1_acc[0].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    # if i % args.print_freq == 0:
    output = (
        "\nEpoch: [{0}][{1}/{2}] \t"
        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        "CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t"
        "SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t"
        "F1 score {f1.val:.4f} ({f1.avg:.4f})\t"
        "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
            epoch,
            i,
            len(train_loader),
            batch_time=batch_time,
            ce_loss=ce_loss_all,
            scl_loss=scl_loss_all,
            f1=f1,
            top1=top1,
        )
    )  

    tsne_targets = torch.cat(tsne_targets, dim=0)
    tsne_f1 = torch.cat(tsne_f1, dim=0)
    tsne_f2 = torch.cat(tsne_f2, dim=0)
    wandb.log(
        {
            "ce_loss_train_avg": ce_loss_all.avg,
            "scl_loss_train_avg": scl_loss_all.avg,
            "top1_train_avg": top1.avg,
            "f1_train_avg": f1.avg,
        },
        step=epoch,
    )
    wandb.log({"epoch": epoch})
    print(output)

    tf_writer.add_scalar("CE loss/train", ce_loss_all.avg, epoch)
    tf_writer.add_scalar("SCL loss/train", scl_loss_all.avg, epoch)
    tf_writer.add_scalar("acc/train_top1", top1.avg, epoch)

    return tsne_targets, tsne_f1, tsne_f2

def validate(
    train_loader,
    val_loader,
    model,
    criterion_ce,
    epoch,
    args,
    tf_writer=None,
    flag="val",
):
    """
    This function evaluates the performance of a given model on a validation set, calculating metrics
    such as cross-entropy loss, accuracy, and F1 score, and logs the results using Weights & Biases.
    
    :param train_loader: A PyTorch DataLoader object for the training set
    :param val_loader: A PyTorch DataLoader object for the validation dataset
    :param model: The neural network model being used for validation
    :param criterion_ce: criterion_ce is the loss function used for training and validation. It is
    typically a cross-entropy loss function that measures the difference between the predicted and
    actual class probabilities
    :param epoch: The current epoch number of the training/validation loop
    :param args: The `args` parameter is a namespace object that contains various arguments and
    hyperparameters passed to the function. It is likely defined in the main script or configuration
    file and contains information such as the learning rate, batch size, number of epochs, etc
    :param tf_writer: tf_writer is a TensorBoard writer object used to log the validation metrics during
    training. It is an optional parameter and can be set to None if TensorBoard logging is not required
    :param flag: The "flag" parameter is a string that specifies whether the function is being called
    for validation or testing. It is used to determine the appropriate dataset loader to use and to
    label the output appropriately, defaults to val (optional)
    :return: a tuple containing the following values:
    - `top1.avg`: average accuracy at top-1
    - `f1.avg`: average F1 score
    - `many_acc_top1`: accuracy at top-1 for classes with many samples
    - `median_acc_top1`: accuracy at top-1 for classes with median samples
    - `low_acc_top1`: accuracy at
    """
    model.eval()
    batch_time = AverageMeter("Time", ":6.3f")
    ce_loss_all = AverageMeter("CE_Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    f1 = AverageMeter("F1", ":.4e")
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    tsne_f1 = []
    tsne_targets = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(tqdm(val_loader)):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)
            feat_mlp, logits, centers = model(inputs, phase="val")

            tsne_f1.append(feat_mlp)
            tsne_targets.append(targets)

            if(args.ce_loss == "WeightedBCE"):
                targets_onehot = torch.nn.functional.one_hot(targets, num_classes=args.cls_num).float()
                ce_loss = criterion_ce(logits, targets_onehot)
            else:
                ce_loss = criterion_ce(logits, targets)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            f1_acc = calc_f1(logits, targets)
            ce_loss_all.update(ce_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)
            f1.update(f1_acc[0].item(), batch_size)

            batch_time.update(time.time() - end)

        # if i % args.print_freq == 0:
        output = (
            "Test: [{0}/{1}]\t"
            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            "CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t"
            "F1 score {f1.val:.4f} ({f1.avg:.4f})\t"
            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                i,
                len(val_loader),
                batch_time=batch_time,
                ce_loss=ce_loss_all,
                f1=f1,
                top1=top1,
            )
        )

        wandb.log(
            {
                "ce_loss_val_avg": ce_loss_all.avg,
                "top1_val_avg": top1.avg,
                "f1_val_avg": f1.avg,
            },
            step=epoch,
        )

        print(output)

        tf_writer.add_scalar("CE loss/val", ce_loss_all.avg, epoch)
        tf_writer.add_scalar("acc/val_top1", top1.avg, epoch)

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(
            preds,
            total_labels,
            train_loader,
            many_shot_thr=args.many_shot_thr,
            low_shot_thr=args.low_shot_thr,
            acc_per_cls=False,
        )
        tsne_targets = torch.cat(tsne_targets, dim=0)
        tsne_f1 = torch.cat(tsne_f1, dim=0)
        return (
            top1.avg,
            f1.avg,
            many_acc_top1,
            median_acc_top1,
            low_acc_top1,
            tsne_targets,
            tsne_f1,
        )

def save_checkpoint(args, state, is_best):
    """
    This function saves a checkpoint of the current state of a model and copies it as the best
    checkpoint if it is the best so far.
    
    :param args: args is a variable that contains various arguments or parameters that are passed to the
    function. These arguments can be used to customize the behavior of the function. In this case, it is
    likely that args contains information about the root log directory, the name of the store, and other
    relevant information needed to save
    :param state: The state parameter is a dictionary that contains the current state of the model,
    optimizer, and other training parameters that need to be saved for resuming training or inference
    later. It typically includes the following keys: 'epoch', 'state_dict', 'optimizer', 'scheduler',
    'best_acc', 'best
    :param is_best: `is_best` is a boolean variable that indicates whether the current checkpoint is the
    best one so far. If `is_best` is `True`, the checkpoint will be saved as the best checkpoint by
    copying it to a new file with the name `best.pth.tar`
    """
    filename = os.path.join(args.root_log, args.store_name, "bcl_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))

class TwoCropTransform:
    # The class `TwoCropTransform` takes two transforms and applies them twice to an input `x`, returning
    # a list of three transformed outputs.
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform2(x)]

def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs + 1)
                / (args.epochs - args.warmup_epochs + 1)
            )
        )
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def calc_f1(output, target):
    """
    It takes the output of the model and the target, and returns the F1 score

    :param output: the output of the model, which is a tensor of shape (batch_size, num_classes)
    :param target: the ground truth labels
    """
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

    return [f1_score(target.cpu(), pred.squeeze(0).cpu(), average="macro")]

def tsne_plot(save_dir, targets, outputs, store_name, phase="train"):
    """
    The function generates a t-SNE plot using the targets and outputs of a model and saves it to a
    specified directory.
    
    :param save_dir: The directory where the t-SNE plot will be saved
    :param targets: The ground truth labels for the data points
    :param outputs: The output of a neural network model, which is a tensor containing the predicted
    values for a batch of input data
    :param store_name: store_name is a string that represents the name of the directory where the t-SNE
    plot will be saved
    :param phase: The phase parameter specifies whether the t-SNE plot is for the training or validation
    phase. It is used to determine the filename of the saved plot, defaults to train (optional)
    """
    print("[INFO] generating t-SNE plot...")
    targets = targets.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=["x", "y"])
    df["targets"] = targets

    plt.rcParams["figure.figsize"] = 10, 10
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sns.scatterplot(
        x="x",
        y="y",
        hue="targets",
        palette=sns.color_palette("deep"),
        data=df,
        marker="o",
        alpha=0.8,
    )

    # plt.scatter(tsne_output[0], tsne_output[1], c=targets, cmap="jet")

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("tSNE 1")
    plt.ylabel("tSNE 2")
    plt.legend()

    if not os.path.exists(os.path.join(save_dir, store_name)):
        os.mkdir(os.path.join(save_dir, store_name))

    filename = ""
    if phase == "val":
        filename = "val_tsne.png"
    elif phase == "train":
        filename = "tsne.png"
    plt.savefig(os.path.join(save_dir, store_name, filename), bbox_inches="tight")
    print("done!")

if __name__ == "__main__":
    main()
