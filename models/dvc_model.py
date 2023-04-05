import torch
import math
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.ops import LowerBound
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.layers import GDN
from compressai.ans import BufferedRansEncoder, RansDecoder
import time
from .ops import ste_round

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class DVC_ChannelARpriors(nn.Module):
    def __init__(self, N=192, M=192, side_input_channels=3, num_slices=8):
        super().__init__()
        self.num_slices = num_slices #each slices == 24

        self.encode_xa = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.encode_side = nn.Sequential(
            conv(side_input_channels, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.decode_x = nn.Sequential(
            deconv(2 * M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.hyper_xa = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.hyper_xs = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + M // self.num_slices * i, M//3 + M//3//self.num_slices* i, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(M//3 + M//3//self.num_slices* i, M//6 + M//6//self.num_slices* i, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(M//6 + M//6//self.num_slices* i, M // self.num_slices, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + M // self.num_slices * i, M//3 + M//3//self.num_slices* i, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(M//3 + M//3//self.num_slices* i, M//6 + M//6//self.num_slices* i, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(M//6 + M//6//self.num_slices* i, M // self.num_slices, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + M // self.num_slices * (i+1), M//3 + M//3//self.num_slices * (i+1), stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(M//3 + M//3//self.num_slices* (i+1), M//6 + M//6//self.num_slices * (i+1), stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(M//6 + M//6//self.num_slices* (i+1), M // self.num_slices, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.N = int(N)
        self.M = int(M)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def aux_loss(self):
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss



    def forward(self, x, side, replace_idx=None):
        #x, side = input[:,:3, :,:], input[:,3:,:,:]
        y = self.encode_xa(x)
        y_shape = y.shape[2:]
        z = self.hyper_xa(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent = self.hyper_xs(z_hat)
        latent_means, latent_scales = latent.chunk(2, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices[:slice_index])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            #print("slice_index:", slice_index, y_slice.size(), mean_support.size(), mu.size())
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            #print("mu:", mu.size())

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            #print("lrp:", lrp_support.size(), lrp.size())

            y_hat_slices.append(y_hat_slice)
            #input()

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        
        fea_side = self.encode_side(side)
        fea_side_likelihoods = self.entropy(fea_side)
        x_hat = self.decode_x(torch.cat((y_hat, fea_side), 1))

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, },
        }

    def entropy(self, y):
        y_shape = y.shape[2:]
        z = self.hyper_xa(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent = self.hyper_xs(z_hat)
        latent_means, latent_scales = latent.chunk(2, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices[:slice_index])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        return y_likelihoods



    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        rv = self.entropy_bottleneck.update(force=force)
        updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def compress(self, x):
        start = time.time()
        y = self.encode_xa(x)
        middle_0 = time.time()
        z = self.hyper_xa(y)
        middle_1 = time.time()
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        middle_2 = time.time()

        latent = self.hyper_xs(z_hat)
        middle_3 = time.time()
        
        latent_means, latent_scales = latent.chunk(2, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        y_shape = y.shape[2:]

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices[:slice_index])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        end = time.time()

        WZ_encode_time = torch.tensor(middle_0-start)
        hyper_encode_time = torch.tensor(middle_1-middle_0)
        z_entropy_time = torch.tensor(middle_2- middle_1)
        hyper_decode_time = torch.tensor(middle_3- middle_2)
        WZ_entropy_time = torch.tensor(end - middle_3)

        total_time = torch.tensor(end - start)
        add_time = WZ_encode_time + hyper_encode_time + z_entropy_time + hyper_decode_time+ WZ_entropy_time
        out = {"Encoder_WZ_encode_time": WZ_encode_time, "Encoder_hyper_encode_time": hyper_encode_time, "Encoder_z_entropy_time": z_entropy_time, 
            "Encoder_hyper_decode_time": hyper_decode_time, "Encoder_WZ_entropy_time": WZ_entropy_time, 
            "Encoder_add_time": add_time, "Encoder_total_time": total_time}

        return y_hat_slices, {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}, out


    def decoder_recon(y_hat_slices, side):
        y_hat = torch.cat(y_hat_slices, dim=1)        
        fea_side = self.encode_side(side)
        x_hat = self.decode_x(torch.cat((y_hat, fea_side), 1))        
        x_hat = x_hat.clamp_(0, 1)
        return {"x_hat": x_hat}


    def decompress(self, strings, shape, side):
        start = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        middle_0 = time.time()
        latent = self.hyper_xs(z_hat)
        middle_1 = time.time()

        latent_means, latent_scales = latent.chunk(2, 1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices[:slice_index])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        middle_2 = time.time()

        fea_side = self.encode_side(side)
        middle_3 = time.time()

        x_hat = self.decode_x(torch.cat((y_hat, fea_side), 1)).clamp_(0, 1)
        end = time.time()

        z_entropy_time = torch.tensor(middle_0-start)
        hyper_decode_time = torch.tensor(middle_1-middle_0)
        WZ_entropy_time = torch.tensor(middle_2- middle_1)
        side_time =torch.tensor( middle_3- middle_2)
        WZ_decode_time = torch.tensor(end - middle_3)

        total_time = torch.tensor(end - start)
        add_time = z_entropy_time + hyper_decode_time+ WZ_entropy_time + side_time + WZ_decode_time
        out = {"Decoder_z_entropy_time": z_entropy_time, "Decoder_hyper_decode_time": hyper_decode_time, "Decoder_WZ_entropy_time": WZ_entropy_time, 
            "Decoder_side_time": side_time, "Decoder_WZ_decode_time": WZ_decode_time, "Decoder_add_time": add_time, "Decoder_total_time": total_time}

        return {"x_hat": x_hat}, out


