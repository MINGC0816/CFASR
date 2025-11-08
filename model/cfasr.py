import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

sys.path.append('./')
sys.path.append('../')
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .model_transformer import FeatureEnhancer, ReasoningTransformer, FeatureEnhancerW2V
from .transformer_v2 import Transformer as Transformer_V2
from .transformer_v2 import InfoTransformer
from .transformer_v2 import PositionalEncoding
from . import torch_distortion
from .dcn import DeformFuser, DSTA
from .language_correction import BCNLanguage
from .gatedfusion import GatedFusion

SHUT_BN = False


class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activation_fn=nn.GELU, dropout_rate=0.):
        super().__init__()

        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer1 = nn.Conv2d(input_dim, hidden_dim, 1, 1)
        self.layer2 = nn.Conv2d(hidden_dim, output_dim, 1, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self, input_dim, qkv_bias=False, qk_scale=None, attn_drop=0., projection_dropout=0.,
                 mode='fc'):
        super().__init__()

        self.query_conv = nn.Conv2d(input_dim, input_dim, 1, 1, bias=qkv_bias)
        self.key_conv = nn.Conv2d(input_dim, input_dim, 1, 1, bias=qkv_bias)
        self.value_conv = nn.Conv2d(input_dim, input_dim, 1, 1, bias=qkv_bias)

        self.horizontal_conv = nn.Conv2d(2 * input_dim, input_dim, (1, 7), stride=1, padding=(0, 7 // 2),
                                         groups=input_dim, bias=False)
        self.vertical_conv = nn.Conv2d(2 * input_dim, input_dim, (7, 1), stride=1, padding=(7 // 2, 0),
                                       groups=input_dim, bias=False)
        self.attention_mlp = FeedForwardLayer(input_dim, input_dim // 4, input_dim * 3)
        self.output_proj = nn.Conv2d(input_dim, input_dim, 1, 1, bias=True)
        self.dropout = nn.Dropout(projection_dropout)
        self.mode = mode

        if mode == 'fc':
            self.horizontal_theta_conv = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1, 1, bias=True),
                nn.BatchNorm2d(input_dim),
                nn.ReLU()
            )
            self.vertical_theta_conv = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1, 1, bias=True),
                nn.BatchNorm2d(input_dim),
                nn.ReLU()
            )
        else:
            self.horizontal_theta_conv = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1, groups=input_dim, bias=False),
                nn.BatchNorm2d(input_dim),
                nn.ReLU()
            )
            self.vertical_theta_conv = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1, groups=input_dim, bias=False),
                nn.BatchNorm2d(input_dim),
                nn.ReLU()
            )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Compute directional theta values
        horizontal_theta = self.horizontal_theta_conv(x)
        vertical_theta = self.vertical_theta_conv(x)

        # Apply convolutions for query, key, and value
        query = self.query_conv(x)
        key = self.key_conv(x)

        # Rotate the query and key based on theta
        query_rotated = torch.cat([query * torch.cos(horizontal_theta), query * torch.sin(horizontal_theta)],
                                  dim=-2).reshape(batch_size, 2 * channels, height, width)
        key_rotated = torch.cat([key * torch.cos(vertical_theta), key * torch.sin(vertical_theta)], dim=-2).reshape(
            batch_size, 2 * channels, height, width)

        # Apply the transformations
        horizontal_features = self.horizontal_conv(query_rotated)
        vertical_features = self.vertical_conv(key_rotated)
        value = self.value_conv(x)

        # Compute attention map
        attention_map = F.adaptive_avg_pool2d(horizontal_features + vertical_features + value, output_size=1)
        attention_map = self.attention_mlp(attention_map).reshape(batch_size, channels, 3).permute(2, 0, 1).softmax(
            dim=0).unsqueeze(-1).unsqueeze(-1)

        # Apply attention weights to combine features
        output = horizontal_features * attention_map[0] + vertical_features * attention_map[1] + value * \
                        attention_map[2]
        output = self.output_proj(output)
        output = self.dropout(output)

        return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class FrequencyAggregationDirectionPerception(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.afm = AFM(dim=dim, ffn_scale=2.0)
        self.norm_afm = norm_layer(dim)
        self.attn = SpatialAttentionModule(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffl = FeedForwardLayer(input_dim=dim, hidden_dim=mlp_hidden_dim, activation_fn=act_layer)
        self.norm_ffl = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.afm(self.norm_afm(x)))
        x = x + self.drop_path(self.attn(self.norm_attn(x)))
        x = x + self.drop_path(self.ffl(self.norm_ffl(x)))
        return x



# 卷积块 (FCB) 模块 FlexConvBlock
class FCB(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


# # 特征增强模块：ContrastAwareAttentionBlock (CAAB)
# #通过注意力机制进行上下文感知的特征增强。与传统卷积不同，该模块根据输入特征的对比性生成注意力权重，用以聚合更显著的区域特征。
class ContrastAwareAttentionBlock(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.v = nn.Linear(dim, dim)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)  # Removed fg/bg distinction
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.input_fcb = nn.Sequential(
            FCB(in_c, dim, kernel_size=3, padding=1),
            FCB(dim, dim, kernel_size=3, padding=1),
        )
        self.output_fcb = nn.Sequential(
            FCB(dim, dim, kernel_size=3, padding=1),
            FCB(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.input_fcb(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)

        attn = self.compute_attention(x, B, H, W, C)  # Use x itself for attention
        x_weighted = self.apply_attention(attn, v_unfolded, B, H, W, C)

        out = self.output_fcb(x_weighted.permute(0, 3, 1, 2))  # (B, C, H, W)
        return out

    def compute_attention(self, feature_map, B, H, W, C):
        # math.ceil()函数用于向上取整
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                     self.kernel_size * self.kernel_size,
                                                     self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted

class ContextualFeatureEnhancer(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

# partial convolution-based feature refinement network
class MaskedFeatureRefiner(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

# self-modulation feature aggregation (SMFA) module
class DynamicFrequencyFusion(nn.Module):
    def __init__(self, dim=36):
        super(DynamicFrequencyFusion, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = ContextualFeatureEnhancer(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        # Introduce a learnable attention map to weigh high/low frequency features
        self.attention_map = nn.Parameter(torch.ones((1, 1, 1, 1)))  # Shape for spatial attention

    def forward(self, f):
        _, _, h, w = f.shape

        # Split the input into high and low-frequency features
        y, x = self.linear_0(f).chunk(2, dim=1)

        # Low-frequency feature processing
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')

        # High-frequency feature processing
        y_d = self.lde(y)

        # Apply adaptive attention to modulate the fusion of high and low-frequency features
        # Attention map will focus more on high-frequency regions
        attention = torch.sigmoid(self.attention_map)

        # Fusion of high-frequency and low-frequency features with weighted attention
        # Enhance high-frequency regions by increasing their weight
        fused_features = (x_l * (1 - attention)) + (y_d * attention)

        return self.linear_2(fused_features)

# Feature modulation block (FMB) AdaptiveFeatureModulator
class AFM(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.dff = DynamicFrequencyFusion(dim)
        self.mfr = MaskedFeatureRefiner(dim, ffn_scale)

    def forward(self, x):
        x = self.dff(F.normalize(x)) + x
        x = self.mfr(F.normalize(x)) + x
        return x


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels, text_channels, kernel_size=3, padding=1, stride=1, num_heads=4, dim=64):
        super(RecurrentResidualBlock, self).__init__()

        self.gru1 = GruBlock(channels + text_channels, channels)
        self.gru2 = GruBlock(channels, channels) # + text_channels

        # self.fadp = FrequencyAggregationDirectionPerception(channels, mlp_ratio=2, qkv_bias=False, qk_scale=None,
        #                           attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')
        #
        # # Initialize ContrastAwareAttentionBlock (CAAB)
        # self.caab = ContrastAwareAttentionBlock(in_c=channels, dim=dim, num_heads=num_heads,
        #                                              kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x, text_emb):
        # Step 1: Process the input with wave MLP
        # aggregated_features = self.fadp(x)
        #
        # # Step 3: Apply ContrastAwareAttentionBlock (CAAB)
        # # Pass the feature x through the CAAB module for enhancement
        # attention_enhanced_features = self.caab(aggregated_features)

        # Step 4: Concatenate enhanced features with text embeddings
        feature_text_concat = torch.cat([x, text_emb], 1)

        # Step 5: Pass through GRU1 (residual update)
        residual = self.gru1(feature_text_concat.transpose(-1, -2)).transpose(-1, -2)

        # Step 6: Add the residual to the input and pass through GRU2 (final residual block)
        return self.gru2(x + residual)


class SGAT(nn.Module):
    def __init__(self, in_features, out_features, head=1, stride=1):
        super(SGAT, self).__init__()
        self.in_features = in_features
        self.hid_features = out_features * 3
        self.head = head
        self.stride = stride
        self.trans = nn.Conv2d(in_channels=self.in_features, out_channels=self.hid_features, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.pad = nn.Unfold(kernel_size=3, padding=1, stride=self.stride)
        self.unfo = nn.Unfold(kernel_size=1, padding=0, stride=stride)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        # result=[]

        x = self.trans(x)
        q, k, v = torch.chunk(x, 3, dim=1)

        b, c, h, w = q.shape

        q = q.reshape(b, self.head, -1, h, w)
        k = k.reshape(b, self.head, -1, h, w)
        v = v.reshape(b, self.head, -1, h, w)

        q = q.view(b * self.head, -1, h, w)
        q = self.unfo(q)
        q = q.view(b, self.head, -1, 1, h // self.stride, w // self.stride).permute(0, 1, 4, 5, 3,
                                                                                    2).contiguous()  # b,head,h,w,1,c//head

        k = k.view(b * self.head, -1, h, w)
        k = self.pad(k)
        k = k.view(b, self.head, -1, 9, h // self.stride, w // self.stride).permute(0, 1, 4, 5, 2,
                                                                                    3).contiguous()  # b,head,h,w,c//head,9

        v = v.view(b * self.head, -1, h, w)
        v = self.pad(v)
        v = v.view(b, self.head, -1, 9, h // self.stride, w // self.stride).permute(0, 1, 4, 5, 3,
                                                                                    2).contiguous()  # b,head,h,w,9,c//head

        att = q @ k
        att = att / math.sqrt(c // self.head)
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        result = att @ v

        result = result.squeeze(-2).permute(0, 1, 4, 2, 3).reshape(b, -1, h // self.stride, w // self.stride)  # stride

        return result


class PAM(nn.Module):
    def __init__(self, in_features, out_features, head=1, stride=1):
        super(PAM, self).__init__()
        self.gama = nn.Parameter(torch.zeros(1))
        self.output = nn.Sequential(
            SGAT(in_features, out_features, head),
            nn.BatchNorm2d(in_features),
            mish()
        )

    def forward(self, x):  # resnet b 2048 16 8
        return (1 - self.gama) * x + self.gama * self.output(x)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pam = PAM(in_channels, in_channels, head=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.pam(x)

        return x


class CFASR(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37,  # 37, #26+26+1 3965
                 out_text_channels=32,
                 triple_clues=False):  # 256 32
        super(CFASR, self).__init__()

        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2),
                    RecurrentResidualBlock(2 * hidden_units, out_text_channels))  # RecurrentResidualBlock

        self.feature_enhancer = None
        # From [1, 1] -> [16, 16]

        self.infoGen = InfoGen(text_emb, out_text_channels)

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)

        self.block_range = [k for k in range(2, self.srb_nums + 2)]
        self.triple_clues = triple_clues

        if self.triple_clues:
            self.lm = BCNLanguage()  # 初始化语言模型
            # self.lm.load_state_dict(torch.load('ckpt/BCN_correct_model.pt'))
            self.dsta_rec = DSTA(hidden_units)
            self.dsta_vis = DSTA(hidden_units)  # 这是一个融合模块，通常用于结合来自不同来源的信息，以提高模型的表现。它可能使用门控机制来选择性地融合信息
            self.dsta_ling = DSTA(hidden_units)
            # self.vis_rec_fuser = DeformFuser(16,hidden_units,hidden_units,4)
            # self.gated = gated(hidden_units)
            self.gated = GatedFusion(hidden_units)
            self.down_conv = nn.Conv2d(hidden_units * 2, hidden_units, 1,
                                       padding=0)  # 此层将输入特征图的通道数从 hidden_units*2 减少到 hidden_units，可能用于特征降维或信息压缩。
            self.infoGen_ling = InfoGen(text_emb,
                                        out_text_channels)  # 用于生成信息的模块，分别针对语言和视觉特征进行处理。这里的 text_emb 表示文本嵌入，可能是通过某种预训练模型获得的。
            self.infoGen_visual = InfoGen(text_emb, 10)
            self.correction_model = BCNLanguage()  # 再次使用 BCNLanguage()，可能用于文本校正或增强。
            self.vis_rec_fuser = DeformFuser(16, hidden_units, hidden_units,
                                             4)  # 一个视觉融合模块，用于处理视觉信息的变形和融合。参数设置可能与输入特征的形状和通道数有关。
        # print("self.block_range:", self.block_range)

    def forward(self, x, text_emb=None, hint_ling=None, hint_vis=None):

        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        block = {'1': self.block1(x)}

        if text_emb is None:
            text_emb = torch.zeros(1, 37, 1, 26).to(x.device)  # 37 or 3965

        spatial_t_emb_gt, pr_weights_gt = None, None
        spatial_t_emb_, pr_weights = self.infoGen(text_emb)  # # ,block['1'], block['1'],

        spatial_t_emb = F.interpolate(spatial_t_emb_, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        B, C, H, W = x.shape
        if self.triple_clues:
            if hint_ling is None:
                hint_ling = torch.zeros_like(text_emb)  # hint_ling 用于语言提示
            if hint_vis is None:
                hint_vis = torch.zeros((B, 6, H, W)).to(
                    x.device)  # hint_vis 用于视觉提示，形状为 (B, 6, H, W)，其中 B 是批量大小，H 和 W 是图像的高度和宽度。

            hint_rec = self.dsta_rec(spatial_t_emb)  # 通过 DSTA 模块处理空间时间嵌入 spatial_t_emb，生成空间提示。
            # hint = spatial_t_emb
            hint_ling, _ = self.infoGen_ling(hint_ling)  # 使用 infoGen_ling 模块处理 hint_ling，并通过双线性插值将其调整到与输入图像相同的空间尺寸。
            hint_ling = F.interpolate(hint_ling, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
            hint_ling = self.dsta_ling(hint_ling)  # 经过插值后的 hint_ling 再次通过 DSTA 模块进行处理，以提取更深层次的特征。
            # hint = hint_ling

            # # The Trident
            # hint_vis_rec, _ = self.infoGen_visual(text_emb)
            # hint_vis_rec = F.interpolate(hint_vis_rec, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
            # hint_vis = self.vis_rec_fuser(torch.cat((hint_vis,hint_vis_rec),1))
            # hint_vis = self.dsta_vis(hint_vis)
            # 使用 GatedFusion 模块将不同来源的信息融合在一起。这里的输入包括通过下卷积层处理后的识别器先验（来自 block['1']）、视觉先验 hint_rec 和语言先验 hint_ling。
            hint = self.gated(self.down_conv(block['1']), hint_rec, hint_ling)

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], hint)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))  #

        output = torch.tanh(block[str(self.srb_nums + 3)])

        self.block = block
        return output


class InfoGen(nn.Module):
    def __init__(
            self,
            t_emb,
            output_size
    ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):
        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class ImFeat2WordVec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImFeat2WordVec, self).__init__()
        self.vec_d = out_channels
        self.vec_proj = nn.Linear(in_channels, self.vec_d)

    def forward(self, x):
        b, c, h, w = x.size()
        result = x.view(b, c, h * w)
        result = torch.mean(result, 2)
        pred_vec = self.vec_proj(result)

        return pred_vec


if __name__ == '__main__':
    # net = NonLocalBlock2D(in_channels=32)
    img = torch.zeros(7, 3, 16, 64)
    embed()
