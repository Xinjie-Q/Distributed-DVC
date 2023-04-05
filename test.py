
import os
import argparse
import json
import math
import sys
import struct
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
import compressai
from compressai.datasets import RawVideoSequence, VideoFormat

from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)

from compressai.zoo.pretrained import load_pretrained
from models import * 
from torch.hub import load_state_dict_from_url
Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]


def collect_videos(rootpath: str) -> List[str]:
    video_files = []
    
    if 'UVG' in rootpath:
        video_files.extend(Path(rootpath).glob(f"1024/*.yuv")) #f"*/*{ext}"
    elif 'MCL_JCV' in rootpath:
        video_files.extend(sorted(Path(rootpath).glob("1024/*.yuv")))
 
    return video_files


# TODO (racapef) duplicate from bench
def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )

def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)

    # sum
    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    # normalize
    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg

def convert_yuv420_to_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    out = to_tensors(frame, device=str(device), max_value=max_val)
    out = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in out), mode="bicubic"  # type: ignore
    )
    return ycbcr2rgb(out)  # type: ignore


def convert_rgb_to_yuv420(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    return yuv_444_to_420(rgb2ycbcr(frame), mode="avg_pool")


def pad(x: Tensor, p: int = 2 ** (4 + 3)) -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(x, padding, mode="replicate")
    return x, padding


def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def compute_metrics_for_frame(
    org_frame: Frame,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,
    index: int = 1,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # YCbCr metrics
    org_yuv = to_tensors(org_frame, device=str(device), max_value=max_val)
    org_yuv = tuple(p.unsqueeze(0).unsqueeze(0) for p in org_yuv)  # type: ignore
    rec_yuv = convert_rgb_to_yuv420(rec_frame)
    for i, component in enumerate("yuv"):
        org = (org_yuv[i] * max_val).clamp(0, max_val).round()
        rec = (rec_yuv[i] * max_val).clamp(0, max_val).round()
        out[f"psnr-{component}"] = 20 * np.log10(max_val) - 10 * torch.log10(
            (org - rec).pow(2).mean()
        )
    out["psnr-yuv"] = (4 * out["psnr-y"] + out["psnr-u"] + out["psnr-v"]) / 6

    # RGB metrics
    org_rgb = convert_yuv420_to_rgb(
        org_frame, device, max_val
    )  # ycbcr2rgb(yuv_420_to_444(org_frame, mode="bicubic"))  # type: ignore
    org_rgb = (org_rgb * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_rgb - rec_frame).pow(2).mean()
    psnr_rgb = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)

    ms_ssim_rgb = ms_ssim(org_rgb, rec_frame, data_range=max_val)
    out.update({"ms-ssim-rgb": ms_ssim_rgb, "mse-rgb": mse_rgb, "psnr-rgb": psnr_rgb})
    return out



def compute_si_metrics_for_frame(
    org_frame: Frame,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # RGB metrics
    org_rgb = convert_yuv420_to_rgb(
        org_frame, device, max_val
    )  # ycbcr2rgb(yuv_420_to_444(org_frame, mode="bicubic"))  # type: ignore
    org_rgb = (org_rgb * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_rgb - rec_frame).pow(2).mean()
    psnr_rgb = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)

    ms_ssim_rgb = ms_ssim(org_rgb, rec_frame, data_range=max_val)
    out.update({"si-ms-ssim-rgb": ms_ssim_rgb, "si-mse-rgb": mse_rgb, "si-psnr-rgb": psnr_rgb})

    return out

def estimate_bits_frame(likelihoods) -> float:
    bpp = sum(
        (torch.log(lkl[k]).sum() / (-math.log(2)))
        for lkl in likelihoods.values()
        for k in ("y", "z")
    )
    return bpp

def compute_bpp(likelihoods, num_pixels: int) -> float:
    bits_per_frame = sum(
        (torch.log(lkl).sum() / (-math.log(2)))
        for lkl in likelihoods.values()
    )
    bpp = bits_per_frame / num_pixels
    return bits_per_frame, bpp

@torch.no_grad()
def eval_model(interpolation_net, BFrameCompressor:nn.Module, IFrameCompressor:nn.Module, 
    sequence: Path, binpath: Path, **args: Any) -> Dict[str, Any]:
    import time
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(BFrameCompressor.parameters()).device
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)
    keep_binaries = args["keep_binaries"]
    num_frames = args["vframes"]
    num_gop = args["GOP"]
    frame_arbitrary = args["frame_arbitrary"]
    with_interpolation = args["with_interpolation"]
    num_pixels = org_seq.height * org_seq.width
    print("frame rate:", org_seq.framerate)
    intra = args["intra"]

    if with_interpolation and not frame_arbitrary:
        frames_idx_list, ref_idx_dict = specific_frame_structure(num_gop)
        reconstructions = []

    f = binpath.open("wb")

    print(f" encoding {sequence.stem}", file=sys.stderr)
    # write original image size
    write_uints(f, (org_seq.height, org_seq.width))
    # write original bitdepth
    write_uchars(f, (org_seq.bitdepth,))
    # write number of coded frames
    write_uints(f, (num_frames,))
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            x_cur = convert_yuv420_to_rgb(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur)

            if i % num_gop == 0:
                start = time.time()
                enc_info = IFrameCompressor.compress(x_cur)
                enc_time = time.time() - start
                write_body(f, enc_info["shape"], enc_info["strings"])
                start = time.time()
                x_rec = IFrameCompressor.decompress(enc_info["strings"], enc_info["shape"])["x_hat"]
                dec_time = time.time() - start
                    
                first_rec = x_rec
                last_key_frame = convert_yuv420_to_rgb(org_seq[i+num_gop], device, max_val)
                last_key_frame, _ = pad(last_key_frame)
                last_enc_info = IFrameCompressor.compress(last_key_frame)
                last_x_rec = IFrameCompressor.decompress(last_enc_info["strings"], last_enc_info["shape"])["x_hat"]
                reconstructions = []
                reconstructions.append(x_rec)
            else:
                if with_interpolation:
                    cur_interpolation_idx = frames_idx_list[i%num_gop-1]
                    left_ref_idx, right_ref_idx = ref_idx_dict[cur_interpolation_idx]
                    if left_ref_idx == 0:
                        left_x_rec = first_rec
                    else:
                        cur_pos_in_frame_idx_list = frames_idx_list.index(left_ref_idx)
                        left_x_rec = reconstructions[cur_pos_in_frame_idx_list+1]
                            
                    if right_ref_idx == num_gop:
                        right_x_rec = last_x_rec
                    else:
                        cur_pos_in_frame_idx_list = frames_idx_list.index(right_ref_idx)
                        right_x_rec = reconstructions[cur_pos_in_frame_idx_list+1]
                    x_cur = convert_yuv420_to_rgb(org_seq[cur_interpolation_idx+(i//num_gop)*num_gop], device, max_val)
                    x_cur, padding = pad(x_cur)
                    start = time.time()
                    y, enc_info = BFrameCompressor.compress(x_cur)
                    enc_time = time.time() - start
                    write_body(f, enc_info["shape"], enc_info["strings"])
                            
                    start = time.time()
                    mid_key = interpolation_net.inference(left_x_rec, right_x_rec, timestep=0.5)
                    x_rec = BFrameCompressor.decompress(enc_info["strings"], enc_info["shape"], mid_key)["x_hat"]
                    dec_time = time.time() - start
                    reconstructions.append(x_rec)
                else:
                    start = time.time()
                    y, enc_info = BFrameCompressor.compress(x_cur)
                    enc_time = time.time() - start
                    write_body(f, enc_info["shape"], enc_info["strings"])
                    start = time.time()
                    mid_key = torch.cat((first_rec, last_x_rec), 1)
                    x_rec = BFrameCompressor.decompress(enc_info["strings"], enc_info["shape"], mid_key)["x_hat"]
                    dec_time = time.time() - start

            x_rec = x_rec.clamp(0, 1)
            if with_interpolation and (i % num_gop != 0):
                metrics = compute_metrics_for_frame(org_seq[cur_interpolation_idx+(i//num_gop)*num_gop], crop(x_rec, padding), device, max_val)
            else:
                metrics = compute_metrics_for_frame(org_seq[i], crop(x_rec, padding), device, max_val)

            if intra or i%num_gop==0:
                metrics["key_encoding_time"] = torch.tensor(enc_time)
                metrics["key_decoding_time"] = torch.tensor(dec_time)
            else:    
                metrics["inter_encoding_time"] = torch.tensor(enc_time)
                metrics["inter_decoding_time"] = torch.tensor(dec_time)

            #print(metrics)
            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)
    f.close()

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    seq_results["bitrate"] = (
        float(filesize(binpath)) * 8 * org_seq.framerate / (num_frames * 1000)
    )
    seq_results["bpp"] = (float(filesize(binpath)) * 8 / (num_frames * num_pixels))


    if not keep_binaries:
        binpath.unlink()

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


def specific_frame_structure(num_gop):
    num_frames = num_gop + 1 #+1-->because add the next key frame
    frames_idx_dict = {3:[1], 5:[2,1,3], 9:[4,2,1,3,6,5,7], 17:[8,4,2,1,3,6,5,7,12,10,9,11,14,13,15], 
        33:[16, 8,4,2,1,3,6,5,7,12,10,9,11,14,13,15,24,20,18,17,19,22,21,23,28,26,25,27,30,29,31]}
    #timestep = 0.5
    #odd_number: -1, +1,  even number 
    ref_idx_dict = {1:[0,2], 3:[2, 4], 5:[4,6], 7:[6, 8], 9:[8, 10], 11:[10, 12], 13:[12, 14], 15:[14, 16],
                    17:[16, 18], 19:[18,20], 21:[20,22], 23:[22,24], 25:[24, 26], 27:[26, 28], 29:[28, 30], 31:[30, 32],
                    6:[4, 8], 10:[8, 12], 12:[8, 16], 14:[12, 16], 18:[16, 20], 20:[16, 24], 22:[20, 24], 24:[16, 32], 
                    26:[24, 28], 28:[24, 32], 30:[28, 32],
                    2:[0, 4], 4:[0, 8], 8:[0, 16], 16:[0, 32]}
    return frames_idx_dict[num_frames], ref_idx_dict


@torch.no_grad()
def eval_model_entropy_estimation(interpolation_net, BFrameCompressor:nn.Module, IFrameCompressor:nn.Module, 
    sequence: Path, **args: Any) -> Dict[str, Any]:
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(IFrameCompressor.parameters()).device
    num_frames = args["vframes"]
    print("video length:{}, frame rate:{}".format(len(org_seq), org_seq.framerate))
    num_pixels = org_seq.height * org_seq.width
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)
    print(f" encoding {sequence.stem}", file=sys.stderr)
    
    num_gop = args["GOP"]
    with_interpolation = args["with_interpolation"]
    frames_idx_list, ref_idx_dict = specific_frame_structure(num_gop)

    with tqdm(total=num_frames) as pbar: #97: 0-96
        for i in range(num_frames):
            x_cur = convert_yuv420_to_rgb(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur)

            if i % num_gop == 0:
                first_key = IFrameCompressor(x_cur)  
                last_key_frame = convert_yuv420_to_rgb(org_seq[i+num_gop], device, max_val)
                last_key_frame, _ = pad(last_key_frame)
                last_key = IFrameCompressor(last_key_frame)  

                x_rec, likelihoods = first_key["x_hat"], first_key["likelihoods"]
                reconstructions = [x_rec]
                current = [x_cur]
                side_info = []
            else:
                cur_interpolation_idx = frames_idx_list[i%num_gop-1]
                left_ref_idx, right_ref_idx = ref_idx_dict[cur_interpolation_idx]
                if left_ref_idx == 0:
                    left_x_rec = first_key["x_hat"]
                else:
                    cur_pos_in_frame_idx_list = frames_idx_list.index(left_ref_idx)
                    left_x_rec = reconstructions[cur_pos_in_frame_idx_list+1]
                            
                if right_ref_idx == num_gop:
                    right_x_rec = last_key["x_hat"]
                else:
                    cur_pos_in_frame_idx_list = frames_idx_list.index(right_ref_idx)
                    right_x_rec = reconstructions[cur_pos_in_frame_idx_list+1]
                x_cur = convert_yuv420_to_rgb(org_seq[cur_interpolation_idx+(i//num_gop)*num_gop], device, max_val)
                x_cur, padding = pad(x_cur)
                if with_interpolation:
                    mid_key = interpolation_net.inference(left_x_rec, right_x_rec, timestep=0.5)
                    side_info.append(mid_key.clamp(0, 1))
                else:
                    mid_key = torch.cat([left_x_rec, right_x_rec], dim=1)

                out = BFrameCompressor(x_cur, mid_key)
                x_rec, likelihoods = out["x_hat"], out["likelihoods"]
                reconstructions.append(x_rec)
                current.append(x_cur)
        
            x_rec = x_rec.clamp(0, 1)
            if i % num_gop != 0:
                org_frame = org_seq[cur_interpolation_idx+(i//num_gop)*num_gop]

            metrics = compute_metrics_for_frame(org_frame, crop(x_rec, padding), device, max_val, i)
            metrics["bitrate"], metrics["bpp"] = compute_bpp(likelihoods, num_pixels)
            if with_interpolation and i%num_gop!=0:
                mid_key = mid_key.clamp(0, 1)
                si_psnr_metrics = compute_si_metrics_for_frame(org_frame, crop(mid_key, padding), device, max_val)
                metrics.update(si_psnr_metrics)

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    seq_results["bitrate"] = float(seq_results["bitrate"]) * org_seq.framerate / 1000
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


def run_inference(
    filepaths,
    interpolation_net, 
    BFrameCompressor: nn.Module, 
    IFrameCompressor: nn.Module, 
    outputdir: Path,
    force: bool = False,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:
    results_paths = []

    for filepath in filepaths:
        sequence_metrics_path = Path(outputdir) / f"{filepath.stem}-{trained_net}.json"
        results_paths.append(sequence_metrics_path)

        if force:
            sequence_metrics_path.unlink(missing_ok=True)
        if sequence_metrics_path.is_file():
            continue

        with amp.autocast(enabled=args["half"]):
            with torch.no_grad():
                if entropy_estimation:
                    metrics = eval_model_entropy_estimation(interpolation_net, BFrameCompressor, IFrameCompressor, filepath, **args)
                else:
                    encode_folder = os.path.join(outputdir, "encoded_files")
                    Path(encode_folder).mkdir(parents=True, exist_ok=True)
                    sequence_bin = Path(encode_folder) / f"{filepath.stem}-{trained_net}.bin" #sequence_metrics_path.with_suffix(".bin")
                    print(sequence_bin)
                    metrics = eval_model(interpolation_net, BFrameCompressor, IFrameCompressor, filepath, sequence_bin, **args)
        with sequence_metrics_path.open("wb") as f:
            output = {
                "source": filepath.stem,
                "name": args["BFrameModel"],
                "description": f"Inference ({description})",
                "results": metrics,
            }
            f.write(json.dumps(output, indent=2).encode())
    results = aggregate_results(results_paths)
    return results

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video compression network evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="sequences directory")
    parser.add_argument("--output", type=str, help="output directory")
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
        default="DVC-ScalePrior",
        choices=models_arch.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("-iq", "--IFrame_quality", type=int, default=4, help='Model quality')
    parser.add_argument("-bq", "--BFrame_quality", type=int, default=1, help='Model quality')
    parser.add_argument("--vframes", type=int, default=96, help='Model quality')
    parser.add_argument(
        "--GOP",
        type=int,
        default=8,
        help="GOP (default: %(default)s)",
    )
    parser.add_argument("--b_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--flownet_model_path", type=str, default="../arXiv2020-RIFE/train_log/RIFE_m_train_log/flownet.pkl", help="Path to a checkpoint")
    parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite previous runs"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--half", action="store_true", help="use AMP")
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--keep_binaries",
        action="store_true",
        help="keep bitstream files in output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    parser.add_argument("--side_input_channels", type=int, default=3, help="use cuda")
    parser.add_argument("--with_interpolation", action="store_true", help='whether use extrapolation network')
    parser.add_argument("--num_slices", type=int, default=8, help="use cuda")
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)


    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )
    filepaths = collect_videos(args.dataset)
    if len(filepaths) == 0:
        print("Error: no video found in directory.", file=sys.stderr)
        raise SystemExit(1)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    #key frame compressor
    IFrameCompressor = models_arch[args.IFrameModel](*cfgs[args.IFrameModel][args.IFrame_quality])
    IFrameCompressor = IFrameCompressor.to(device)
    url = model_urls[args.IFrameModel][args.metric][args.IFrame_quality]
    checkpoint = load_state_dict_from_url(url, progress=True, map_location=device)
    checkpoint = load_pretrained(checkpoint)
    IFrameCompressor.load_state_dict(checkpoint)
    IFrameCompressor.eval()


    if args.b_model_path:
        if args.with_interpolation:
            interpolation_net = VideoInterpolationNet(args, arbitrary=True)
            print("Loading Video Interpolation model:", args.flownet_model_path)
            interpolation_net.load_model(args.flownet_model_path)
            interpolation_net.device(device)
            interpolation_net.eval()
        else:
            interpolation_net = None

        #wyner-ziv encoder and decoder
        BFrameCompressor = models_arch[args.BFrameModel](*cfgs[args.BFrameModel][args.BFrame_quality], args.side_input_channels, num_slices=args.num_slices)
        print(args.BFrameModel, BFrameCompressor.num_slices)
        BFrameCompressor = BFrameCompressor.to(device)
        print("Loading B frame model: ", args.b_model_path)
        checkpoint = torch.load(args.b_model_path, map_location=device)
        BFrameCompressor.load_state_dict(checkpoint["state_dict"])
        BFrameCompressor.update(force=True)
        BFrameCompressor.eval()
    else:
        interpolation_net = None
        BFrameCompressor = None

    # create output directory
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    args_dict = vars(args)

    trained_net = f"{args.BFrameModel}-{args.metric}-{description}"


    metrics = run_inference(filepaths, interpolation_net, BFrameCompressor, IFrameCompressor, 
        outputdir, trained_net=trained_net, description=description, **args_dict,)
    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": f"{args.BFrameModel}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{args.BFrameModel}-{description}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    #print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
