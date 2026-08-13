import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# --------------------------------------------------------------------------- #
# Entropy-mode names accepted by ResNet(entropy_mode=...).
#
# 'shannon' and 'unique' are the original two and behave exactly as before; the
# rest are additive alternatives used by the entropy-comparison experiment
# (mlep/experiments/entropy.py). entropy_mode may also be a LIST of these names, in
# which case one map per mode is stacked on the channel dim -- with mode 0's
# channels bit-identical to the single-mode config, so combined stacks stay
# clean additive tests. Covered by tests/test_entropy_modes.py.
#
#   shannon        -(1/K) sum_i log2(c_i / K)                (base 2)
#   unique         number of distinct values in the window
#   renyi_<a>      log2(sum_g p_g^a) / (1 - a),  a > 0, a != 1
#   tsallis_<q>    (1 - sum_g p_g^q) / (q - 1),  q > 0, q != 1
#   perm           permutation (ordinal-pattern) entropy of the window; see
#                  ResNet._perm_entropy_map for the exact pattern definition.
#
# On a 2x2 window all of these except perm are five-valued -- four pixels can
# only be equal in five patterns -- so they are read off a lookup table instead
# of being computed (ResNet._entropy_2x2_table). Same numbers, ~65x cheaper than
# the sort-based general path, which is what makes the non-Shannon families
# affordable to train at all.
# --------------------------------------------------------------------------- #
def parse_entropy_mode(mode):
    """'shannon' | 'unique' | 'renyi_2' | 'tsallis_0.5' | 'perm' -> (kind, param).

    Raises ValueError on anything else, so a typo in a config is a crash rather
    than a silent fall-back to shannon."""
    if not isinstance(mode, str):
        raise ValueError(f"entropy_mode entries must be strings, got {mode!r}")
    if mode in ('shannon', 'unique', 'perm'):
        return mode, None
    for kind in ('renyi', 'tsallis'):
        prefix = kind + '_'
        if mode.startswith(prefix):
            try:
                param = float(mode[len(prefix):])
            except ValueError:
                raise ValueError(f"bad {kind} order in entropy_mode {mode!r}")
            if param <= 0 or param == 1.0:
                raise ValueError(f"{kind} order must be > 0 and != 1, got {param}")
            return kind, param
    raise ValueError(
        f"unknown entropy_mode {mode!r}; expected 'shannon', 'unique', 'perm', "
        f"'renyi_<a>' or 'tsallis_<q>'")


def entropy_max(kind, param, K):
    """Largest value the given entropy can take over K symbols -- the divisor
    used by normalize_entropy. For perm the alphabet is the ordinal-pattern code
    (6 realisable orderings of 3 distinct values), not K."""
    if kind == 'unique':
        return float(K)
    if kind == 'perm':
        return math.log2(6.0)
    if kind == 'tsallis':
        return (1.0 - K ** (1.0 - param)) / (param - 1.0)
    return math.log2(K)          # shannon, renyi (both peak at log2 K)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 window_sizes=(2,), scales=(1.0, 0.5, 0.25), entropy_mode='shannon',
                 use_rearrange=True, rearrange_block_size=2, normalize_entropy=False,
                 window_align='resize', texture_split=False, texture_patch_size=16,
                 texture_split_mode='concat', color_entropy=False):
        super(ResNet, self).__init__()

        # --- Multi-scale local-entropy (MIE) front-end configuration ---------- #
        # The network turns the input image into a stack of local-entropy maps
        # (one per window-size x scale) and feeds that stack to conv1. The number
        # of input channels therefore depends on how many windows/scales we use:
        #   in_channels = len(window_sizes) * len(scales) * 3 (RGB)
        # ...doubled when texture_split reassembles the image into a rich and a
        # poor canvas and stacks both ('concat'); see _split_texture_canvases.
        # Defaults reproduce the released paper setting exactly (2x2 window, a
        # 3-scale pyramid, shannon entropy, no texture split) -> 9 channels, so
        # the pretrained resnet50 weights load unchanged.
        self.window_sizes = list(window_sizes)
        self.scales = list(scales)
        self.entropy_mode = entropy_mode
        # entropy_mode is a single name (the original behaviour) or a list of
        # names to stack. entropy_modes is always the list form; every method
        # below takes the resolved single mode as an argument, so a str
        # entropy_mode reaches exactly the code it always did.
        self.entropy_modes = ([entropy_mode] if isinstance(entropy_mode, str)
                              else list(entropy_mode))
        if not self.entropy_modes:
            raise ValueError("entropy_mode must name at least one entropy")
        for _m in self.entropy_modes:
            _kind, _ = parse_entropy_mode(_m)
            if _kind == 'perm' and min(self.window_sizes) < 3:
                raise ValueError("entropy_mode 'perm' needs window_sizes >= 3 "
                                 "(ordinal patterns are order 3); got "
                                 f"{self.window_sizes}")
        self.use_rearrange = use_rearrange
        self.rearrange_block_size = rearrange_block_size
        self.normalize_entropy = normalize_entropy

        # --- PatchCraft-style rich/poor texture separation (opt-in) ----------- #
        # texture_split=True splits the image into non-overlapping
        # texture_patch_size x texture_patch_size patches, ranks them by texture
        # diversity and reassembles the busy half into one "rich" canvas and the
        # flat half into a "poor" canvas. Each canvas goes through the SAME
        # entropy front-end; texture_split_mode says how the two stacks are then
        # combined:
        #   'concat' -- stack both, conv1 sees 2x the channels. The network gets
        #               the two texture populations as separate inputs.
        #   'diff'   -- feed (rich - poor), keeping the channel count. This is
        #               what PatchCraft itself uses (the rich/poor *contrast* is
        #               the generator fingerprint) and it is channel-matched to
        #               the corresponding non-split config, so a win cannot be
        #               explained away by conv1 having more input width.
        # Adapted from Zhong et al., "Exploring Texture Patch for Efficient
        # AI-generated Image Detection". Default False -> nothing about the
        # released 9-channel path changes.
        self.texture_split = texture_split
        self.texture_patch_size = texture_patch_size
        if texture_split_mode not in ('concat', 'diff'):
            raise ValueError(f"texture_split_mode must be 'concat' or 'diff', "
                             f"got {texture_split_mode!r}")
        self.texture_split_mode = texture_split_mode

        # --- Joint-colour entropy (opt-in) ------------------------------------ #
        # The existing front-end is ALREADY per-channel, not grayscale:
        # _entropy_2x2_shannon keeps the channel dim throughout, so the 9 default
        # channels are (R,G,B) x 3 scales. What it does NOT have is any CROSS-channel
        # statistic -- the three maps are independent marginals, and a window can be
        # maximally entropic in R, in G and in B while containing only two distinct
        # COLOURS (e.g. the 2x2 pattern [(0,1,0),(1,0,1),(0,1,0),(1,0,1)] has 2
        # distinct values per channel but 2 distinct RGB triples, not 8).
        # color_entropy adds the entropy of the joint RGB-triple distribution, where
        # two pixels count as the same symbol only if they agree in EVERY channel:
        #   'joint'      -- append it, so conv1 sees 4 maps per (scale, window)
        #                   instead of 3. Strictly more information than the
        #                   baseline, and the first 3 channels are bit-identical to
        #                   it, so the pair is a clean additive test.
        #   'joint_only' -- feed ONLY the joint map (1 per scale/window, 3 channels
        #                   total at the default pyramid). Channel-cheaper than the
        #                   baseline, so it isolates whether the joint statistic
        #                   carries the signal on its own rather than riding along.
        # Why it might matter for the robustness problem: JPEG/WebP subsample chroma
        # (4:2:0) and quantise the two chroma planes far harder than luma, so
        # recompression damages cross-channel structure and per-channel structure by
        # different amounts. A marginal-only front-end cannot see that difference.
        # Default False -> nothing about the released 9-channel path changes.
        if color_entropy is None:
            color_entropy = False
        if color_entropy not in (False, 'joint', 'joint_only'):
            raise ValueError(f"color_entropy must be False, 'joint' or 'joint_only', "
                             f"got {color_entropy!r}")
        if color_entropy and any(parse_entropy_mode(m)[0] == 'perm'
                                 for m in self.entropy_modes):
            # The joint map's alphabet is the RGB triple; permutation entropy is
            # defined on the ORDER of scalar values, which a triple does not have.
            raise ValueError("color_entropy is not defined for entropy_mode 'perm'")
        self.color_entropy = color_entropy

        # How to reconcile the differently-sized maps that different window
        # sizes produce (a w-window over an H-wide input yields H-w+1 columns):
        #   'resize' -- bilinear-resample every map onto the first map's grid.
        #               The original behaviour, and the default.
        #   'pad'    -- replicate-pad every map out to the largest map's size.
        #               Exact, see the note in forward(). Requires all window
        #               sizes to share a parity so the pad is a whole number.
        if window_align not in ('resize', 'pad'):
            raise ValueError(f"window_align must be 'resize' or 'pad', got {window_align!r}")
        if window_align == 'pad':
            parities = {w % 2 for w in self.window_sizes}
            if len(parities) > 1:
                raise ValueError(
                    "window_align='pad' needs every window size to be the same "
                    "parity (all even or all odd) so that (w - min_w) / 2 is a "
                    f"whole number of pixels; got {self.window_sizes}.")
        self.window_align = window_align
        # 'concat' feeds two canvases' worth of maps; 'diff' subtracts one from
        # the other, so the stack keeps the un-split width.
        n_canvas = 2 if (self.texture_split and self.texture_split_mode == 'concat') else 1
        # Maps per (canvas, scale, window): the 3 per-channel marginals, plus or
        # replaced by the joint-colour map (see color_entropy above).
        per_map = {False: 3, 'joint': 4, 'joint_only': 1}[self.color_entropy]
        # ...once per entropy mode when several are stacked (len == 1 -> unchanged).
        per_map *= len(self.entropy_modes)
        in_channels = n_canvas * len(self.window_sizes) * len(self.scales) * per_map

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Only layer1/layer2 are used, so the feature width entering fc1 is
        # 128 * block.expansion (512 for Bottleneck/resnet50, 128 for
        # BasicBlock/resnet18). Hardcoding 512 crashed resnet18 -- derive it.
        self.fc1 = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    # patch-based entropy
    def _entropy_2x2_shannon(self, x):
        """
        Independently calculate local color entropy for each channel of the input image x.
        The local block size is 2x2, allowing for overlapping calculations.

        This is the exact discrete categorisation used by the released paper /
        pretrained weights (note: the "three same" case uses 0.8, a rounding of
        the true Shannon value 0.8113). It is kept as a fast special case for the
        default 2x2 shannon setting; other window sizes / entropy modes go
        through the general `_patch_entropy` path below.
        Color entropy is assigned based on the following five situations:
        1. All pixels are the same: entropy value is 0.0
        2. Three pixels are the same, one is different: entropy value is 0.81
        3. Two pairs of pixels are the same: entropy value of 1.0
        4. Two pixels are the same, the other two are different: entropy value is 1.5
        5. All pixels are different: entropy value is 2.0
        """

        batch, channels, H, W = x.size()

        # Extract 2x2 sliding windows with a stride of 1
        patches = x.unfold(2, 2, 1).unfold(3, 2, 1)  # (batch, channels, H-1, W-1, 2, 2)
        # print("patches shape: ",patches.shape)
        # reshaping as (batch, channels, (H-1)*(W-1), 4)
        patches = patches.contiguous().view(batch, channels, (H-1)*(W-1), 4)
        # print("After contiguous patches shape: ",patches.shape)
        # Extract 4 pixel values from each 2x2 block
        p0 = patches[:, :, :, 0]
        p1 = patches[:, :, :, 1]
        p2 = patches[:, :, :, 2]
        p3 = patches[:, :, :, 3]
        # print("p0 shape: ", p0.shape)
        # Calculate masks for various situations
        all_same = (p0 == p1) & (p1 == p2) & (p2 == p3)  # All four pixel values are the same.
        # print("all_same shape: ",all_same.shape)
        
        three_same = ((p0 == p1) & (p1 == p2) & (p2 != p3)) | \
                     ((p0 == p1) & (p1 == p3) & (p2 != p3)) | \
                     ((p0 == p2) & (p2 == p3) & (p1 != p2)) | \
                     ((p1 == p2) & (p2 == p3) & (p0 != p1))  # Three pixels are the same, one is different
        
        
        two_pairs = ((p0 == p1) & (p2 == p3) & (p1 != p2)) | \
                    ((p0 == p2) & (p1 == p3) & (p0 != p1)) | \
                    ((p0 == p3) & (p1 == p2) & (p0 != p1))  # Two pairs of pixels are identical.
        
        two_same_two_diff = ((p0 == p1) & (p2 != p3) & (p1 != p2) & (p1 != p3)) | \
                            ((p0 == p2) & (p1 != p3) & (p0 != p1) & (p0 != p3)) | \
                            ((p0 == p3) & (p1 != p2) & (p0 != p1) & (p0 != p2)) | \
                            ((p1 == p2) & (p0 != p3) & (p1 != p0) & (p1 != p3)) | \
                            ((p1 == p3) & (p0 != p2) & (p1 != p0) & (p1 != p2)) | \
                            ((p2 == p3) & (p0 != p1) & (p2 != p0) & (p2 != p1))  # Two pixels are the same, the other two are different.
        
        all_diff = (p0 != p1) & (p0 != p2) & (p0 != p3) & \
                   (p1 != p2) & (p1 != p3) & (p2 != p3)  # All pixels are different.
        
        # Initialize category index.
        entropy_image = torch.zeros_like(p0, dtype=torch.float, device=x.device)
        entropy_image = torch.where(all_same, torch.tensor(0.0, device=x.device), entropy_image)
        # entropy_image = torch.where(three_same, torch.tensor(0.81, device=x.device), entropy_image)
        entropy_image = torch.where(three_same, torch.tensor(0.8, device=x.device), entropy_image)
        entropy_image = torch.where(two_pairs, torch.tensor(1.0, device=x.device), entropy_image)
        entropy_image = torch.where(two_same_two_diff, torch.tensor(1.5, device=x.device), entropy_image)
        entropy_image = torch.where(all_diff, torch.tensor(2.0, device=x.device), entropy_image)
        
        entropy_image = entropy_image.view(batch, channels, H-1, W-1)

        return entropy_image

    # The five ways four pixels can share values, written as each element's group
    # size -- exactly the `count` tensor _patch_entropy reconstructs by sorting.
    # A 2x2 window admits no others, so ANY entropy of a 2x2 window takes at most
    # five values and is fully determined by which of these five patterns holds.
    _PARTITIONS_2X2 = ((4., 4., 4., 4.),      # all four equal
                       (3., 3., 3., 1.),      # three equal, one different
                       (2., 2., 2., 2.),      # two pairs
                       (2., 2., 1., 1.),      # one pair, two singletons
                       (1., 1., 1., 1.))      # all four different
    # ...and the partitions are told apart by counting equal pairs among the six
    # unordered pairs: 6 / 3 / 2 / 1 / 0 respectively. 4 and 5 are unreachable.
    _PAIRS_2X2 = (6, 3, 2, 1, 0)

    def entropy_table_2x2(self, kind, param):
        """The five values `kind` takes on a 2x2 window, laid out for a lookup
        indexed by the number of equal pairs (see _PAIRS_2X2); the two impossible
        slots hold nan so a miscount cannot pass silently.

        Values come from _entropy_from_group_counts, the same function the
        general path uses, so the table cannot drift away from it -- this is a
        speed shortcut, not a second definition. Shannon is the exception: it
        keeps the released lookup (0.8 for the three-same case, a rounding of
        log2(3) - 2/3) so pretrained weights stay valid."""
        table = [float('nan')] * 7
        counts = torch.tensor(self._PARTITIONS_2X2)                   # (5, 4)
        if kind == 'shannon':
            vals = [0.0, 0.8, 1.0, 1.5, 2.0]                          # legacy
        elif kind == 'unique':
            vals = [1.0, 2.0, 2.0, 3.0, 4.0]
        else:
            vals = self._entropy_from_group_counts(counts, 4, kind, param).tolist()
        for n_pairs, v in zip(self._PAIRS_2X2, vals):
            table[n_pairs] = v
        return table

    def _entropy_2x2_table(self, x, kind, param):
        """Fast 2x2 path for any entropy: classify the window, then look the
        value up.

        The general path sorts every window and rebuilds its run lengths, which
        for w=2 is a lot of work to distinguish five cases. Counting how many of
        the six pixel pairs are equal identifies the case with six comparisons
        and one gather, which is what the released Shannon path effectively does
        too -- this just does it for every mode."""
        batch, channels, H, W = x.size()
        p = x.unfold(2, 2, 1).unfold(3, 2, 1).contiguous()
        p = p.view(batch, channels, (H - 1) * (W - 1), 4)
        p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        n_pairs = ((p0 == p1).long() + (p0 == p2).long() + (p0 == p3).long()
                   + (p1 == p2).long() + (p1 == p3).long() + (p2 == p3).long())
        table = torch.tensor(self.entropy_table_2x2(kind, param),
                             dtype=torch.float32, device=x.device)
        return table[n_pairs].view(batch, channels, H - 1, W - 1)

    def _entropy_from_group_counts(self, count, K, kind, param):
        """Entropy of a value distribution given each ELEMENT's group size.

        `count[..., i]` is the number of elements sharing element i's value, so
        averaging a per-element term over the K elements weights each group g by
        p_g = c_g / K -- i.e. mean_i f(c_i) == sum_g p_g f(c_g). That identity is
        what lets every mode below share one code path:

            shannon    -sum_g p_g log2 p_g          = mean_i -log2(c_i / K)
            renyi_a    log2(sum_g p_g^a) / (1 - a)  with sum_g p_g^a = sum_i (c_i/K)^a / c_i
            tsallis_q  (1 - sum_g p_g^q) / (q - 1)  same sum, natural (non-log) units
        """
        p = count / K
        if kind == 'shannon':
            return -(torch.log2(p)).mean(dim=-1)
        # sum over GROUPS from a per-ELEMENT tensor: divide each term by its own
        # group size so a group of c identical entries contributes exactly once.
        s = (p.pow(param) / count).sum(dim=-1)
        if kind == 'renyi':
            return torch.log2(s) / (1.0 - param)
        return (1.0 - s) / (param - 1.0)          # tsallis

    def _patch_entropy(self, patches, K, mode=None):
        """General per-patch entropy over the last dim (K = window_size**2 pixels).

        `patches`: (batch, channels, num_patches, K).
        Returns (batch, channels, num_patches).

        `mode` defaults to self.entropy_mode (the original single-mode
        behaviour); the entropy experiment passes one resolved mode at a time.

        - 'unique'      -> number of distinct pixel values in the patch.
        - 'shannon'     -> Shannon entropy (base 2) of the patch's value
          distribution, computed without materialising a KxK equality matrix:
          sort the patch, count each value's group size via cummax/cummin on the
          run boundaries, then  H = -(1/K) * sum_i log2(count_i / K).
          This matches -sum_g p_g log2 p_g and was cross-checked against a
          brute-force reference over 20k random patches (max err ~3e-15).
        - 'renyi_<a>' / 'tsallis_<q>' -> the same group counts, different
          functional (see _entropy_from_group_counts). Both -> shannon as a -> 1.
        """
        kind, param = parse_entropy_mode(self.entropy_mode if mode is None else mode)
        sorted_p, _ = torch.sort(patches, dim=-1)
        change = torch.ones_like(sorted_p, dtype=torch.bool)
        change[..., 1:] = sorted_p[..., 1:] != sorted_p[..., :-1]

        if kind == 'unique':
            return change.sum(dim=-1).float()

        # recover each element's group size from the sorted run structure.
        idx = torch.arange(K, device=patches.device, dtype=torch.int32).expand_as(change)
        zeros = torch.zeros_like(idx)
        big = torch.full_like(idx, K)
        # start[i] = index where element i's run begins
        start = torch.cummax(torch.where(change, idx, zeros), dim=-1).values
        # nxt[i] = index where the next run begins (K for the last run)
        cand = torch.where(change, idx, big)
        suffix_min = torch.flip(torch.cummin(torch.flip(cand, dims=[-1]), dim=-1).values, dims=[-1])
        nxt = big.clone()
        nxt[..., :-1] = suffix_min[..., 1:]
        count = (nxt - start).float()
        return self._entropy_from_group_counts(count, K, kind, param)

    def _perm_entropy_map(self, x, w):
        """Permutation (ordinal-pattern) entropy over each w x w neighbourhood.

        Returns (batch, channels, H-w+1, W-w+1), same grid as _entropy_map.

        Bandt-Pompe permutation entropy needs a DISTRIBUTION of ordinal patterns,
        and a single window read as one sequence gives exactly one pattern (zero
        entropy, always). So the pattern population inside a window is its
        order-3 sub-sequences: every horizontal triplet of every row plus every
        vertical triplet of every column -- 2*w*(w-2) patterns for a w x w window
        (6 for 3x3, 16 for 4x4). Each triplet (a, b, c) is coded by its three
        pairwise comparisons, and the entropy is the Shannon entropy (base 2) of
        the code histogram. Codes are 3 bits, of which 6 of the 8 values are
        realisable by distinct reals; ties (flat regions -- exactly what JPEG and
        blur produce) collapse onto the '>' -> False side, which is consistent
        rather than arbitrary and keeps flat windows at entropy 0.

        Unlike shannon/renyi/tsallis this reads the ORDER of the values rather
        than their multiset, so it survives any monotone intensity change and
        responds to local smoothing instead of to quantisation.
        """
        p = x.unfold(2, w, 1).unfold(3, w, 1)        # (B, C, Hn, Wn, w, w), a view
        counts = torch.zeros(p.shape[:4] + (8,), dtype=torch.float32, device=x.device)
        for i in range(w - 2):
            for a, b, c in ((p[..., i, :], p[..., i + 1, :], p[..., i + 2, :]),
                            (p[..., :, i], p[..., :, i + 1], p[..., :, i + 2])):
                code = ((a > b).long() + 2 * (a > c).long() + 4 * (b > c).long())
                counts.scatter_add_(-1, code, torch.ones_like(code, dtype=counts.dtype))
        total = counts.sum(dim=-1, keepdim=True)
        pr = counts / total
        # 0 * log2(0) := 0; clamp only inside the log, the zero factor kills the term.
        return -(pr * torch.log2(pr.clamp_min(1e-12))).sum(dim=-1)

    def _entropy_map(self, x, w, mode=None):
        """Local-entropy map for a single window size `w` (stride 1, overlapping).

        Returns (batch, channels, H-w+1, W-w+1). Uses the exact legacy 2x2
        categorisation for the default (w=2, shannon) case so pretrained weights
        stay valid, and the general path otherwise. Optionally normalised to
        [0, 1] (divide by the mode's maximum, see entropy_max)."""
        mode = self.entropy_mode if mode is None else mode
        kind, param = parse_entropy_mode(mode)
        if kind == 'perm':
            e = self._perm_entropy_map(x, w)
        elif w == 2 and kind == 'shannon':
            e = self._entropy_2x2_shannon(x)
        elif w == 2:
            # unique / renyi / tsallis on a 2x2 window: five possible values, so
            # the sort-based general path is pure overhead (see _entropy_2x2_table).
            e = self._entropy_2x2_table(x, kind, param)
        else:
            batch, channels, H, W = x.size()
            K = w * w
            patches = x.unfold(2, w, 1).unfold(3, w, 1).contiguous()
            Hn, Wn = patches.size(2), patches.size(3)
            patches = patches.view(batch, channels, Hn * Wn, K)
            e = self._patch_entropy(patches, K, mode).view(batch, channels, Hn, Wn)

        if self.normalize_entropy:
            e = e / entropy_max(kind, param, w * w)
        return e

    def _joint_entropy_map(self, x, w, mode=None):
        """Local entropy of the JOINT colour distribution, one map for all channels.

        Returns (batch, 1, H-w+1, W-w+1) -- a single map, not one per channel:
        the symbol alphabet is the RGB *triple*, so two pixels are the same symbol
        only when they agree in every channel. This is the cross-channel statistic
        `_entropy_map` structurally cannot express, since it treats each channel as
        an independent marginal.

        Implemented via the K x K same-colour matrix rather than by packing the
        triple into one number: packing needs the exact quantisation levels, and by
        the time the tensor reaches the model it has been through ImageNet
        `Normalize`, so the levels are per-channel affine images of 0..255 that this
        module has no way to recover. Pairwise equality is exact regardless of what
        affine map was applied, and K = w*w is tiny (4 for the default 2x2).

        `entropy_mode` is honoured: 'unique' counts distinct colours, 'shannon'
        gives -sum_g p_g log2 p_g over the colour groups. Note the 2x2 shannon case
        uses the TRUE value here (0.8113 for three-alike) rather than the legacy 0.8
        rounding baked into `_entropy_2x2_shannon`; the marginal channels keep the
        legacy value, so the two differ by 0.011 on one of five categories. That is
        deliberate -- the legacy rounding exists only to stay bit-compatible with the
        released weights, and this map is new so nothing depends on it.
        """
        kind, param = parse_entropy_mode(self.entropy_mode if mode is None else mode)
        batch, channels, H, W = x.size()
        K = w * w
        patches = x.unfold(2, w, 1).unfold(3, w, 1).contiguous()
        Hn, Wn = patches.size(2), patches.size(3)
        patches = patches.view(batch, channels, Hn * Wn, K)

        # same[b, n, i, j] iff pixels i and j of window n share every channel value.
        # AND-ed channel by channel rather than as one (B, C, N, K, K) comparison
        # that is then .all(1)-reduced: same result, but the peak buffer is 2 of
        # these instead of C+1 (12.7 MiB vs 38 MiB at batch 16 / crop 224 / w=2).
        same = patches[:, 0].unsqueeze(-1) == patches[:, 0].unsqueeze(-2)
        for c in range(1, channels):
            same &= patches[:, c].unsqueeze(-1) == patches[:, c].unsqueeze(-2)

        if kind == 'unique':
            # Count each colour once by counting only its FIRST pixel: pixel i is
            # first iff no j < i shares its colour.
            idx = torch.arange(K, device=x.device)
            earlier = idx.view(-1, 1) > idx.view(1, -1)          # strictly-lower mask
            e = (~(same & earlier).any(dim=-1)).sum(dim=-1).float()
        else:
            # count[i] = size of pixel i's colour group, so averaging -log2(count/K)
            # over the K pixels weights each group by its own size == sum_g p_g.
            # renyi / tsallis reuse the same counts (see _entropy_from_group_counts).
            count = same.sum(dim=-1).float()
            e = self._entropy_from_group_counts(count, K, kind, param)

        e = e.view(batch, 1, Hn, Wn)
        if self.normalize_entropy:
            e = e / entropy_max(kind, param, K)
        return e

    # Divide and shuffle.
    def random_rearrange_blocks(self, x, block_size=2):
        """
        Split the input image x into 8x8 small blocks and arrange them randomly.
        If the image size cannot be divided by 8, the excess part will be cropped.

        parameter:
            x (torch.Tensor): input image，with the shape of (batch, channels, H, W).
            block_size (int): size of blocks，the default is 8.

        return:
            rearranged_x (torch.Tensor): The randomly arranged image, with the shape of (batch, channels, H', W').
        """
        batch, channels, H, W = x.size()

        # Crop the image so that its size can be evenly divided by block_size.
        H_cropped = (H // block_size) * block_size
        W_cropped = (W // block_size) * block_size
        x_cropped = x[:, :, :H_cropped, :W_cropped]

        # Split the image into small blocks.
        blocks = x_cropped.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        num_blocks_H = blocks.size(2)
        num_blocks_W = blocks.size(3)
        
        # Reshape as (batch, channels, num_blocks, block_size, block_size).
        blocks = blocks.contiguous().view(batch, channels, num_blocks_H * num_blocks_W, block_size, block_size)
        
        # Fix the permutation only at eval time (deterministic, matches the
        # released weights). During training we must NOT fix it: the original
        # `torch.manual_seed(99)` here ran on every forward, which both froze the
        # shuffle to a single pattern and clobbered the global RNG used for
        # batching/augmentation. Guarding it lets training see real random shuffles.
        #
        # The seed lives in a LOCAL generator rather than torch.manual_seed(99):
        # reseeding the global RNG from inside a forward pass is a side effect on
        # the whole process -- it silently resets any other torch randomness the
        # caller is relying on, once per eval batch. A local generator gives the
        # same deterministic shuffle without reaching outside this function.
        #
        # During training generate the permutation directly on x's device: keeps
        # indices and data co-located, avoiding a per-image CPU<->GPU sync (a real
        # cost on a 4090). At eval we keep CPU randperm so the deterministic
        # shuffle stays byte-identical to the released-weights behaviour.
        if self.training:
            perm_device, gen = x.device, None
        else:
            perm_device = None
            gen = torch.Generator()
            gen.manual_seed(99)

        for i in range(batch):
            permuted_indices = torch.randperm(num_blocks_H * num_blocks_W,
                                              device=perm_device, generator=gen)
            blocks[i] = blocks[i, :, permuted_indices, :, :]

        # Reorder the blocks into (batch, channels, num_blocks_H, num_blocks_W, block_size, block_size)
        blocks = blocks.view(batch, channels, num_blocks_H, num_blocks_W, block_size, block_size)

        # Splicing blocks to reconstruct images
        rearranged_x = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        rearranged_x = rearranged_x.view(batch, channels, H_cropped, W_cropped)

        return rearranged_x

    # --- PatchCraft-style rich / poor texture separation ---------------------- #
    def _texture_diversity(self, x, p):
        """Per-patch texture diversity, PatchCraft-style (Zhong et al.).

        Sums the absolute pixel fluctuation in four directions -- horizontal,
        vertical, diagonal, anti-diagonal -- inside each non-overlapping pxp
        patch. Flat sky scores ~0; grass / hair / fabric scores high.

        Computed on the mean over RGB rather than per channel. ImageNet
        Normalize divides the channels by 0.229 / 0.224 / 0.225, which are within
        2% of each other, and every term below is a difference of two pixels in
        the SAME channel, so the per-channel means cancel outright. A
        per-channel-then-summed measure is therefore ~3x this one with a ~2%
        reweighting -- the same ranking for 1/3 of the work.

        `x`: (batch, channels, H, W), H and W already whole multiples of p.
        Returns (batch, (H//p) * (W//p)), patches in raster order.
        """
        g = x.mean(dim=1, keepdim=True)              # (batch, 1, H, W)
        q = g.unfold(2, p, p).unfold(3, p, p)        # (batch, 1, nh, nw, p, p)
        batch, _, nh, nw = q.shape[:4]
        # Unfolding into patches BEFORE differencing is the point: it keeps the
        # four sums from reaching across a patch boundary. contiguous() copies
        # one channel's worth of image (~6 MB at batch 32 / 224px), not the whole
        # tensor.
        q = q.contiguous().view(batch, nh * nw, p, p)
        return ((q[:, :, :, 1:]   - q[:, :, :, :-1]).abs().sum(dim=(-2, -1)) +
                (q[:, :, 1:, :]   - q[:, :, :-1, :]).abs().sum(dim=(-2, -1)) +
                (q[:, :, 1:, 1:]  - q[:, :, :-1, :-1]).abs().sum(dim=(-2, -1)) +
                (q[:, :, 1:, :-1] - q[:, :, :-1, 1:]).abs().sum(dim=(-2, -1)))

    def _split_texture_canvases(self, x, p):
        """Reassemble x into a texture-RICH and a texture-POOR canvas.

        Both canvases hold exactly half the patches and therefore have IDENTICAL
        H and W -- which is the whole point. The two entropy stacks can then be
        concatenated with no bilinear resampling, and not resampling the
        high-frequency entropy structure is exactly what window_align='pad'
        exists to achieve (see the note in forward()).

        Grid arithmetic, with nh = H//p and nw = W//p patches:
            nh even  -> split by ROWS,    canvas grid (nh/2, nw)  -- lossless
            nw even  -> split by COLUMNS, canvas grid (nh, nw/2)  -- lossless
            both odd -> drop the last patch COLUMN, then split by columns.
        Preferring a lossless branch matters: at cropSize 224 with p=32 the grid
        is 7x7 and the trim throws away 14.3% of the crop, which would confound
        any comparison against the un-split baseline. p=16 is lossless at 128,
        224 AND 256, which is why it is the default. Excess rows/columns that do
        not fill a whole patch are cropped, exactly as random_rearrange_blocks
        already does.

        Ties -- e.g. a flat image where every patch scores 0 -- are broken by
        raster index: sorting with stable=True keeps equal elements in input
        order, so the split is reproducible and identical on CPU and CUDA.
        """
        batch, channels, H, W = x.size()
        nh, nw = H // p, W // p
        if nh >= 2 and nh % 2 == 0:
            gh, gw = nh // 2, nw
        elif nw >= 2 and nw % 2 == 0:
            gh, gw = nh, nw // 2
        elif nw >= 2:
            nw -= 1                      # both odd: trim one patch column
            gh, gw = nh, nw // 2
        elif nh >= 2:
            nh -= 1
            gh, gw = nh // 2, nw
        else:
            raise ValueError(
                f"texture_patch_size={p} leaves a {H // p}x{W // p} patch grid on a "
                f"{H}x{W} input; need at least 2 patches along one axis to split.")

        x = x[:, :, :nh * p, :nw * p]
        num, half = nh * nw, (nh * nw) // 2

        # descending -> index 0 is the most textured patch.
        order = torch.argsort(self._texture_diversity(x, p), dim=1,
                              descending=True, stable=True)            # (batch, num)
        patches = (x.unfold(2, p, p).unfold(3, p, p)
                    .contiguous().view(batch, channels, num, p, p))
        ordered = torch.gather(
            patches, 2, order.view(batch, 1, num, 1, 1)
                             .expand(batch, channels, num, p, p))

        def canvas(sel):
            # Same reassembly permutation as random_rearrange_blocks: lay the
            # patches out in raster order over a gh x gw grid.
            return (sel.view(batch, channels, gh, gw, p, p)
                       .permute(0, 1, 2, 4, 3, 5)
                       .contiguous().view(batch, channels, gh * p, gw * p))

        return canvas(ordered[:, :, :half]), canvas(ordered[:, :, half:])

    # def save_intermediate_image(self, x, folder):
    #     """
    #     Save the image to the specified folder.
    #     """
    #     # Normalize to [0, 1] range if necessary
    #     x = x.clamp(0, 2.0) / 2.0  # Ensure the values are in range [0, 1]

    #     # Convert from tensor to image and save
    #     save_dir = Path(folder)
    #     save_dir.mkdir(parents=True, exist_ok=True)

    #     # Iterate through the batch dimension if it's > 1
    #     for i in range(x.size(0)):
    #         # Get the i-th image from the batch (assuming 3 channels)
    #         img = x[i]

    #         # Convert tensor to PIL image and save it
    #         img_name = f"image_{i}.png"  # You can customize the name
    #         img_path = save_dir / img_name

    #         # Save the image
    #         save_image(img, img_path)

    #         print(f"Saved intermediate image to {img_path}")

    def forward(self, x):

        # ---------MIE start here---------
        # Optional PatchCraft-style texture separation. It runs BEFORE the block
        # shuffle and on the original image, because texture diversity is only
        # meaningful on spatially coherent patches: random_rearrange_blocks
        # scatters 2x2 blocks across the whole image, which would leave every
        # patch with the same (image-average) diversity and collapse the split
        # into a raster-order tie-break -- an experiment that measures nothing
        # while looking perfectly healthy.
        if self.texture_split:
            canvases = self._split_texture_canvases(x, self.texture_patch_size)
        else:
            canvases = (x,)

        # Divide and shuffle (optional). Applied to each canvas SEPARATELY, so a
        # block can never migrate between the rich and the poor canvas. Note both
        # canvases get the same permutation at eval, since random_rearrange_blocks
        # builds a fresh Generator().manual_seed(99) per call -- deterministic and
        # harmless, not a bug.
        if self.use_rearrange:
            canvases = tuple(self.random_rearrange_blocks(c, self.rearrange_block_size)
                             for c in canvases)

        # For every canvas, every scale (downsample-then-upsample; scale 1.0 ==
        # identity) and every window size, compute a local-entropy map. With the
        # defaults (one canvas, scales [1, .5, .25], window [2], shannon) this
        # reproduces the original three-map / 9-channel behaviour exactly, in the
        # same channel order.
        # Channel order is CANVAS-major, then scale, then window, then RGB:
        #   idx = canvas*(S*Wn*3) + scale*(Wn*3) + window*3 + rgb
        # so under 'concat' the first len(scales)*len(window_sizes)*3 channels are
        # exactly the stack the equivalent un-split config would produce on the
        # rich canvas, and the rest is the poor canvas.
        # In 'pad' mode every map is grown out to the smallest window's map size.
        # Note the maps are stride-1, so a w-window map is H-w+1 wide and each of
        # its cells sits one pixel from the next -- every window size samples the
        # SAME unit lattice, they just start further in and stop earlier. Cell i
        # of the w map is centred on input coordinate i + (w-1)/2, which is where
        # cell i + (w-w_min)/2 of the w_min map sits. So padding by
        # d = (w - w_min) / 2 on each side both restores the size
        # ((H-w+1) + 2d == H-w_min+1) and lands every cell on its true position.
        # No interpolation is involved, which is the whole point: resampling
        # would blur the high-frequency entropy structure the model relies on.
        w_min = min(self.window_sizes)
        per_canvas = []
        # `xin`, not `x`: forward() rebinds x to the conv1 output further down, and
        # the scale loop reads xin.shape[2:] as its upsample target. Reusing the
        # name would silently upsample the second canvas to the first one's shape.
        for xin in canvases:
            feats = []
            for s in self.scales:
                if s == 1.0:
                    xs = xin
                else:
                    down = F.interpolate(xin, scale_factor=s, mode='bilinear', align_corners=False)
                    xs = F.interpolate(down, size=xin.shape[2:], mode='bilinear', align_corners=False)
                for w in self.window_sizes:
                    # One group of maps per entropy mode, concatenated in the
                    # order entropy_mode lists them. With a single mode (a str,
                    # the original case) this is exactly the old single map, so
                    # combined stacks keep mode 0's channels bit-identical to the
                    # corresponding single-mode config.
                    per_mode = []
                    for m in self.entropy_modes:
                        em = self._entropy_map(xs, w, m)
                        if self.color_entropy:
                            # Appended AFTER the marginals, so channels 0..2 of every
                            # group stay bit-identical to the color_entropy=False stack.
                            # Both go through the same window_align padding below.
                            ej = self._joint_entropy_map(xs, w, m)
                            em = ej if self.color_entropy == 'joint_only' \
                                else torch.cat([em, ej], dim=1)
                        per_mode.append(em)
                    e = per_mode[0] if len(per_mode) == 1 else torch.cat(per_mode, dim=1)
                    if self.window_align == 'pad':
                        d = (w - w_min) // 2
                        if d:
                            # replicate, not zeros: a zero border would read as a
                            # perfectly flat (zero-entropy) frame around every image,
                            # which is a strong artefact the model would latch onto.
                            e = F.pad(e, (d, d, d, d), mode='replicate')
                    feats.append(e)
            per_canvas.append(feats)

        if self.texture_split and self.texture_split_mode == 'diff':
            # The rich/poor entropy CONTRAST, as in PatchCraft. Same channel count
            # as the un-split config, so the comparison is capacity-matched.
            feats = [rich - poor for rich, poor in zip(per_canvas[0], per_canvas[1])]
        else:
            feats = [f for canvas_feats in per_canvas for f in canvas_feats]

        # Different window sizes yield slightly different map sizes; resize them
        # all to the first map's size before concatenating on the channel dim.
        # Under 'pad' -- and across canvases, which are the same size by
        # construction -- the sizes already agree, so this loop is a no-op.
        target = feats[0].shape[-2:]
        feats = [f if f.shape[-2:] == target
                 else F.interpolate(f, size=target, mode='bilinear', align_corners=False)
                 for f in feats]
        concat_result = torch.cat(feats, dim=1)

        # Subsequent convolution. conv1.in_channels was sized to match this stack.
        x = self.conv1(concat_result)

        # ---------MIE end here---------

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
