import torch
import torch.nn as nn

# --- 1D Depthwise Convolution ---
# Applies a separate filter to each input channel for 1D data.
# Achieved by setting groups = in_channels.
# The number of output channels is typically a multiple of the input channels (depth_multiplier * in_channels).
# For simplicity, here out_channels = in_channels (depth_multiplier = 1).
class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseConv1d, self).__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,  # Each channel gets its own filter
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels, # Key for depthwise convolution
            bias=bias
        )

    def forward(self, x):
        return self.depthwise_conv(x)

# --- 1D Pointwise Convolution ---
# Applies a 1x1 convolution (in the temporal/sequence dimension) to change the channel dimension for 1D data.
# It mixes information across channels.
class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(PointwiseConv1d, self).__init__()
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, # Key for pointwise convolution
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        return self.pointwise_conv(x)

# --- 1D Depthwise Separable Convolution ---
# Consists of a 1D Depthwise Convolution followed by a 1D Pointwise Convolution.
# This factorization significantly reduces computation and parameters for 1D data.
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise_conv = DepthwiseConv1d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.pointwise_conv = PointwiseConv1d(
            in_channels=in_channels, # Output of depthwise is input to pointwise
            out_channels=out_channels,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
