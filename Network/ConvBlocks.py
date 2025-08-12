from torch import nn
import copy

# The most fucking basic building block of convolutional neural network
class ConvolutionalBlock(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = False, padding_mode = "replicate", activation = None, norm = None, groups = 1):
    super(ConvolutionalBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias = bias, padding_mode = padding_mode, groups = groups)
    self.bn = copy.deepcopy(norm) or nn.BatchNorm2d(output_channels, eps = 0.001, momentum = 0.03, affine = True)
    self.activation = copy.deepcopy(activation) or nn.SiLU()
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.activation(x)
    return x

class DepthwiseConvolutionalBlock(nn.Module): # by default, 3x3 convolution but channels are processed independently
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=False, padding_mode="replicate", activation=None, norm=None):
        super(DepthwiseConvolutionalBlock, self).__init__()
        self.block = ConvolutionalBlock(
            input_channels=channels,
            output_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            activation=activation,
            groups=channels,
            norm=norm
        )

    def forward(self, x):
        return self.block(x)

class PointwiseConvolutionalBlock(nn.Module): # Applies 1x1 convolution, effectively dense layer
    def __init__(self, input_channels, output_channels, bias=False, activation=None, norm=None):
        super(PointwiseConvolutionalBlock, self).__init__()
        self.block = ConvolutionalBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
            activation=activation,
            norm=norm,
            groups=1
        )

    def forward(self, x):
        return self.block(x)

# THIS IS SO SMART WHY DIDN'T UNIVERSITY TEACH ME AAAAAAA
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, bias = False, padding_mode = "replicate", activation = None, norm = None, num_depthwise = 1):
        super(DepthwiseSeparableConvolution, self).__init__()
        depthwise_convs = [ DepthwiseConvolutionalBlock(
            channels=input_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            norm=norm,
            activation=activation
        ) for _ in range(num_depthwise) ]

        self.depthwise = nn.Sequential(*depthwise_convs)
        self.pointwise = PointwiseConvolutionalBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            bias=bias,
            norm=norm,
            activation=activation
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Bottleneck(nn.Module):
  def __init__(self, input_channels, bottleneck_channels, output_channels, repeats = 1, bias = False, activation = None, shortcut = False, norm = None):
    super(Bottleneck, self).__init__()
    self.downscaling = PointwiseConvolutionalBlock(input_channels, bottleneck_channels, bias = bias, norm=norm) if input_channels != bottleneck_channels else nn.Identity()
    processing_layers = [ ConvolutionalBlock(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=bias, activation = activation, norm=norm) for _ in range(repeats) ]
    self.processing_layers = nn.Sequential(*processing_layers)
    self.upscaling = PointwiseConvolutionalBlock(bottleneck_channels, output_channels, bias = bias, activation=nn.Identity(), norm=norm) if bottleneck_channels != output_channels else nn.Identity()
    self.shortcut = shortcut
    self.activation = activation or nn.SiLU()
  def forward(self, x):
    residual = x
    x = self.downscaling(x)
    x = self.processing_layers(x)
    x = self.upscaling(x)
    if self.shortcut: x = x + residual
    return self.activation(x)

class CSP1_X(nn.Module):
  def __init__(self, input_channels, working_channels, output_channels, X = 1, bias = False, activation = None, norm = None):
    super(CSP1_X, self).__init__()
    self.rescaling_in = PointwiseConvolutionalBlock(input_channels, working_channels, bias = bias, norm=norm) if input_channels != working_channels else nn.Identity()
    branch_A = [ ConvolutionalBlock(working_channels, working_channels, kernel_size=3, padding=1, bias=bias, activation=activation, norm=norm) ]
    branch_A.extend( [ Bottleneck(working_channels, working_channels, working_channels, repeats=2, bias=bias, activation=activation, shortcut=True, norm=norm) for _ in range(X) ] ) # Residual blocks, effectively
    self.branch_A = nn.Sequential(*branch_A)
    branch_B = [nn.Conv2d(working_channels, working_channels, kernel_size=1, padding=0, bias=bias)]
    self.branch_B = nn.Sequential(*branch_B)
    self.bn = nn.BatchNorm2d(2*working_channels, eps = 0.001, momentum = 0.03, affine = True)
    self.activation = activation if activation else nn.SiLU()
    self.rescaling_out = PointwiseConvolutionalBlock(2*working_channels, output_channels, bias = bias, activation=activation, norm=norm)
  def forward(self, x):
    x = self.rescaling_in(x)
    output_A = self.branch_A(x)
    output_B = self.branch_B(x)
    x = torch.cat([output_A, output_B], dim=1)
    x = self.bn(x)
    x = self.activation(x)
    x = self.rescaling_out(x)
    return x

class CSP2_X(nn.Module):
  def __init__(self, input_channels, working_channels, output_channels, X = 1, bias = False, activation = None, norm = None):
    super(CSP2_X, self).__init__()
    self.rescaling_in = PointwiseConvolutionalBlock(input_channels, working_channels, bias = bias, norm=norm) if input_channels != working_channels else nn.Identity()
    branch_A = [ ConvolutionalBlock(working_channels, working_channels, kernel_size=3, padding=1, bias=bias, activation=activation, norm=norm) ]
    branch_A.extend( [ ConvolutionalBlock(working_channels, working_channels, kernel_size=3, padding=1, bias=bias, activation=activation, norm=norm) for _ in range(2*X) ] ) # Residual blocks, effectively
    self.branch_A = nn.Sequential(*branch_A)
    branch_B = [nn.Conv2d(working_channels, working_channels, kernel_size=1, padding=0, bias=bias)]
    self.branch_B = nn.Sequential(*branch_B)
    self.bn = nn.BatchNorm2d(2*working_channels, eps = 0.001, momentum = 0.03, affine = True)
    self.activation = activation if activation else nn.SiLU()
    self.rescaling_out = PointwiseConvolutionalBlock(2*working_channels, output_channels, bias = bias, activation=activation, norm=norm)
  def forward(self, x):
    x = self.rescaling_in(x)
    output_A = self.branch_A(x)
    output_B = self.branch_B(x)
    x = torch.cat([output_A, output_B], dim=1)
    x = self.bn(x)
    x = self.activation(x)
    x = self.rescaling_out(x)
    return x
