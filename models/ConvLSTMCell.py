import torch.nn as nn
import torch
import numpy as np


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, **kwargs):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        if 'custom_init' in kwargs.keys() and kwargs['custom_init'] is not None:
            eval(kwargs['custom_init'])(self.conv.weight)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def normal_custom(w):
    fan_out, fan_in = w.shape[:2]
    std = float(fan_in + fan_out) / float(fan_in * fan_out)
    return _no_grad_normal_(w, 0., std)


def normal_custom1(w):
    fan_out, fan_in = w.shape
    std = np.sqrt(float(fan_in + fan_out) / float(fan_in * fan_out))
    return _no_grad_normal_(w, 0., std)

def normal_custom2(w):
    fan_out, fan_in = w.shape
    std = np.sqrt(1 / np.abs(fan_in - fan_out))
    return _no_grad_normal_(w, 0., std)