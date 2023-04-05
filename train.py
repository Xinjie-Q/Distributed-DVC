import argparse
import math
import random
import shutil
import sys
import os
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from lib.utils import get_output_folder, Vimeo, AverageMeter, save_checkpoint, CustomDataParallel
from models import * 
from torch.hub import load_state_dict_from_url
from compressai.zoo.pretrained import load_pretrained 


def configure_optimizers(net, interpolation_net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )
    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    #dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    if args.flow_finetune and args.with_interpolation:
        optimizer = optim.Adam([{'params': (params_dict[n] for n in sorted(parameters)), 'lr':args.learning_rate},
            {'params': interpolation_net.flownet.parameters(), 'lr': args.flow_learning_rate}]
            )
    else:
        optimizer = optim.Adam(
            (params_dict[n] for n in sorted(parameters)),
            lr=args.learning_rate,
            )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def train_one_epoch(IFrameCompressor, BFrameCompressor, interpolation_net, criterion, train_dataloader, 
    optimizer, aux_optimizer, epoch, iterations, clip_max_norm, args):
    IFrameCompressor.train()
    BFrameCompressor.train()
    if args.with_interpolation:
        interpolation_net.train()
    device = next(BFrameCompressor.parameters()).device
    
    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')

    metric_dB_name = 'psnr' if args.metric == "mse" else "ms_ssim_db"
    metric_name = "mse_loss" if args.metric == "mse" else "ms_ssim_loss"
    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')

    train_dataloader = tqdm(train_dataloader)
    print('Train epoch:', epoch)
    for i, images in enumerate(train_dataloader):
        rand_num = random.randint(3, len(images))
        images_index = random.sample(range(len(images)), rand_num)
        images_index.sort(reverse=False) #升序
        images = [images[idx].to(device) for idx in images_index]
        num_p = len(images)-1

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        for imgidx in range(num_p):
            if imgidx == 0:
                # I frame compression.
                first_key = IFrameCompressor(images[0])
                last_key = IFrameCompressor(images[-1])
            else:
                # B frame compression
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                if args.with_interpolation:
                    mid_key = interpolation_net.inference(first_key["x_hat"], last_key["x_hat"], timestep=((imgidx)/num_p))
                    if not args.flow_finetune:
                        mid_key = mid_key.detach()
                else:
                    mid_key = torch.cat((first_key["x_hat"], last_key["x_hat"]), 1)

                out = BFrameCompressor(images[imgidx], mid_key)

                out_criterion = criterion(out, images[imgidx], args.lmbda)
                out_criterion["loss"].backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(BFrameCompressor.parameters(), args.clip_max_norm)  # mxh add
                optimizer.step()
                out_aux_loss = BFrameCompressor.aux_loss()
                out_aux_loss.backward()
                aux_optimizer.step()

                loss.update(out_criterion["loss"].item())
                bpp_loss.update(out_criterion["bpp_loss"].item())
                aux_loss.update(out_aux_loss.item())
                metric_loss.update(out_criterion[metric_name].item())
                metric_dB.update(out_criterion[metric_dB_name].item())
                iterations += 1

        train_dataloader.set_description('[{}/{}]'.format(i, len(train_dataloader)))
        train_dataloader.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "iterations": iterations}

    return out


def test_epoch(epoch, test_dataloader, IFrameCompressor, BFrameCompressor, interpolation_net, criterion, args):
    IFrameCompressor.eval()
    BFrameCompressor.eval()
    if args.with_interpolation:
        interpolation_net.eval()
    device = next(BFrameCompressor.parameters()).device

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    metric_dB_name = 'psnr' if args.metric == "mse" else "ms_ssim_db"
    metric_name = "mse_loss" if args.metric == "mse" else "ms_ssim_loss"
    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')   

    test_dataloader = tqdm(test_dataloader)
    with torch.no_grad():
        for i, images in enumerate(test_dataloader):
            rand_num = random.randint(3, len(images))
            images_index = random.sample(range(len(images)), rand_num)
            images_index.sort(reverse=False) #升序
            images = [images[idx].to(device) for idx in images_index]
            num_p = len(images)-1

            for imgidx in range(num_p):
                if imgidx == 0:
                    # I frame compression.
                    first_key = IFrameCompressor(images[0])
                    last_key = IFrameCompressor(images[-1])               
                else:
                    if args.with_interpolation:
                        mid_key = interpolation_net.inference(first_key["x_hat"], last_key["x_hat"], timestep=((imgidx)/num_p))
                    else:
                        mid_key = torch.cat((first_key["x_hat"], last_key["x_hat"]), 1)

                    out = BFrameCompressor(images[imgidx], mid_key)

                    out_criterion = criterion(out, images[imgidx], args.lmbda)  

                    loss.update(out_criterion["loss"].item())
                    bpp_loss.update(out_criterion["bpp_loss"].item())
                    aux_loss.update(BFrameCompressor.aux_loss().item())
                    metric_loss.update(out_criterion[metric_name].item())
                    metric_dB.update(out_criterion[metric_dB_name].item())

                    test_dataloader.set_description('[{}/{}]'.format(i, len(test_dataloader)))
                    test_dataloader.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                                metric_dB_name:metric_dB.avg})


    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg}

    return out

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-im",
        "--IFrameModel",
        default="mbt2018",
        choices=models_arch.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-bm",
        "--BFrameModel",
        default="DVC-Hyperprior",
        choices=models_arch.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("-iq", "--IFrame_quality", type=int, default=4, help='Model quality')
    parser.add_argument("-bq", "--BFrame_quality", type=int, default=1, help='Model quality')
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--flow_learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0,
        help="Bit-rate distortion parameter (default: %(default)s)",
    ) #0.0018; λ2 = 0.0035; λ3 = 0.0067; λ4 = 0.0130,  λ5 =0.025

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save IFrameCompressor to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=1, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=5.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--i-model-path", type=str, default="/home/xzhangga/.cache/torch/hub/checkpoints/mbt2018-4-456e2af9.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--b_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--flownet_model_path", type=str, default="./flownet_model/RIFE_m_train_log/flownet.pkl", help="Path to a checkpoint")
    parser.add_argument("--metric", type=str, default="ms-ssim", help="metric: mse, ms-ssim")
    parser.add_argument("--flow_finetune", action="store_true", help='whether flownet is finetuned')#default: False
    parser.add_argument("--with_interpolation", action="store_true", help='whether use extrapolation network')
    parser.add_argument("--use_pretrained_bmodel", action="store_true", help='use pretrained high rate model to train low rate model')
    parser.add_argument("--side_input_channels", type=int, default=3)
    parser.add_argument("--num_slices", type=int, default=10)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)


    train_dataset = Vimeo(args.dataset, is_training=True, crop_size=args.patch_size)
    test_dataset = Vimeo(args.dataset, is_training=False, crop_size=args.patch_size)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    #key frame compressor
    IFrameCompressor = models_arch[args.IFrameModel](*cfgs[args.IFrameModel][args.IFrame_quality])
    IFrameCompressor = IFrameCompressor.to(device)
    for p in IFrameCompressor.parameters():
        p.requires_grad = False

    url = model_urls[args.IFrameModel][args.metric][args.IFrame_quality]
    checkpoint = load_state_dict_from_url(url, progress=True, map_location=device)
    checkpoint = load_pretrained(checkpoint)
    IFrameCompressor.load_state_dict(checkpoint)

    #wyner-ziv encoder and decoder
    BFrameCompressor = models_arch[args.BFrameModel](*cfgs[args.BFrameModel][args.BFrame_quality], args.side_input_channels)
    BFrameCompressor = BFrameCompressor.to(device)
  
    if args.with_interpolation:
        interpolation_net = VideoInterpolationNet(args, arbitrary=True)
        interpolation_net.load_model(args.flownet_model_path)
        if not args.flow_finetune:
            interpolation_net.freeze_model()
        interpolation_net.device(device)
        model_name = "NDVC_Interpolation"  
    else:
        interpolation_net = None
        model_name = "NDVC_WO_Interpolation"

    if args.metric == "mse":
        criterion = DVCLoss()
    else:
        criterion = DVC_MS_SSIM_Loss(device, size_average=True, max_val=1)

    optimizer, aux_optimizer = configure_optimizers(BFrameCompressor, interpolation_net, args)
    patience = 5 if args.flow_finetune and args.with_interpolation else 10
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=patience, factor=0.2)
    
    last_epoch = 0
    last_iterations = 0
    best_loss = float("inf")
    use_previous_bmode_path = False
    
    if args.b_model_path:
        print("Loading B frame model: ", args.b_model_path)
        checkpoint = torch.load(args.b_model_path, map_location=device)
        if args.use_pretrained_bmodel:
            print("load pretrained model")
            BFrameCompressor.load_state_dict(checkpoint["state_dict"])
        else:
            print("load pretrained model and optimizer!")
            BFrameCompressor.load_state_dict(checkpoint["state_dict"])
       
            last_epoch = checkpoint["epoch"] + 1
            last_iterations = checkpoint["iterations"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_b_model_path = os.path.join(os.path.split(args.b_model_path)[0], 'ckpt.best.pth.tar')
            best_loss = torch.load(best_b_model_path)["loss"] #checkpoint["loss"]
           
            use_previous_bmode_path = True

    if args.cuda and torch.cuda.device_count() > 1:
        IFrameCompressor = CustomDataParallel(IFrameCompressor)
        BFrameCompressor = CustomDataParallel(BFrameCompressor)

    stage = 2 if args.flow_finetune else 1
    if use_previous_bmode_path:
        log_dir = os.path.split(args.b_model_path)[0]
    else:
        log_dir = get_output_folder('./checkpoints/{}/{}/{}/stage{}'.format(args.metric, model_name, args.BFrameModel, stage), 'train')

    print(log_dir)
    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    writer = SummaryWriter(log_dir)
    
    metric_dB_name = 'psnr' if args.metric == "mse" else "ms_ssim_db"
    metric_name = "mse_loss" if args.metric == "mse" else "ms_ssim_loss"
    iterations = last_iterations
    #val_loss = test_epoch(0, test_dataloader, IFrameCompressor, BFrameCompressor, interpolation_net, criterion, args)
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(IFrameCompressor, BFrameCompressor, interpolation_net, criterion, train_dataloader,
            optimizer, aux_optimizer, epoch, iterations, args.clip_max_norm, args)
        val_loss = test_epoch(epoch, test_dataloader, IFrameCompressor, BFrameCompressor, interpolation_net, criterion, args)
     
        writer.add_scalar('train/loss', train_loss["loss"], epoch)
        writer.add_scalar('train/bpp_loss', train_loss["bpp_loss"], epoch)
        writer.add_scalar('train/aux_loss', train_loss["aux_loss"], epoch)  

        writer.add_scalar('val/loss', val_loss["loss"], epoch)        
        writer.add_scalar('val/bpp_loss', val_loss["bpp_loss"], epoch)
        writer.add_scalar('val/aux_loss', val_loss["aux_loss"], epoch)

        writer.add_scalar('train/'+metric_dB_name, train_loss[metric_dB_name], epoch)
        writer.add_scalar('train/'+metric_name, train_loss[metric_name], epoch)
        writer.add_scalar('val/'+metric_name, val_loss[metric_name], epoch)
        writer.add_scalar('val/'+metric_dB_name, val_loss[metric_dB_name], epoch)

        iterations = train_loss["iterations"]
        loss = val_loss["loss"]
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "state_dict": BFrameCompressor.state_dict(),
                    "loss": loss,
                    "bpp": val_loss["bpp_loss"],
                    metric_dB_name: val_loss[metric_dB_name],
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best, log_dir
            )

            if args.flow_finetune:
                interpolation_net.save_model(log_dir, is_best)

if __name__ == "__main__":
    main(sys.argv[1:])
