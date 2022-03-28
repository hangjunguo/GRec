from torch import nn
import torch
import torch.functional as F
import torch.nn.functional as F2
import math
import numpy as np
import utils


class GRecArchi(nn.Module):

    def __init__(self, model_para):
        super(GRecArchi, self).__init__()
        self.item_size = model_para['item_size']
        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']

        self.embed_en = EmbeddingLayer(self.item_size, self.residual_channels)
        self.encoder = NextItNet_ED(model_para, causal=False)
        self.embed_de = EmbeddingLayer(self.item_size, self.residual_channels)
        self.projector = Projector(self.residual_channels, 2 * self.residual_channels)
        self.decoder = NextItNet_ED(model_para, causal=True)
        self.final = ConvDilate(self.residual_channels, self.item_size, kernel_size=1)

    def forward(self, itemseq_input_en, itemseq_input_de, positions=None, test=False):
        self.test = test
        output = self.embed_en(itemseq_input_en)
        output = self.encoder(output)  # encoder
        output = output.add(self.embed_de(itemseq_input_de))  # agg
        output = self.projector(output)  # projector
        output = self.decoder(output)  # decoder [batch_size, seq_len, dilated_channels = self.residual_channels = embed_size]
        output = self.masked_logits(output, positions)

        return output

    def masked_logits(self, input_tensor, positions):
        """Get logits for the masked LM and probs for the last item (if test)."""

        residual_channels = input_tensor.size(-1)
        if self.test:
            logits = self.final(F2.relu(input_tensor)[:, -1:, :])  ## FC, d dim to n dim
            logits_2D = logits.reshape(-1, self.item_size)  # [batch_size * 1, item_size)
            m = nn.Softmax(dim=1)
            logits_2D = m(logits_2D)
        else:
            input_tensor = self.gather_ids(input_tensor, positions)  # retrieve the hidden vectors of masked positions
            logits = self.final(F2.relu(input_tensor))  ## FC, d dim to n dim
            logits_2D = logits.reshape(-1, self.item_size)  # [batch_size * masked_length, item_size)

        return logits_2D

    def gather_ids(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""

        batch_size = sequence_tensor.size(0)
        seq_length = sequence_tensor.size(1)
        masked_length = positions.size(1)
        width = sequence_tensor.size(2)

        flat_offsets = torch.LongTensor(torch.arange(0, batch_size) * seq_length).view(-1, 1).to(positions.device.type)

        flat_positions = torch.add(positions, flat_offsets).view(-1)
        flat_sequence_tensor = sequence_tensor.view(batch_size * seq_length, width)
        output_tensor = torch.index_select(flat_sequence_tensor, 0, flat_positions)  # [batch_size*masked_length, width]
        output_tensor = output_tensor.view(-1, masked_length, width)
        return output_tensor  # [batch_size, masked_length, width]


class EmbeddingLayer(nn.Module):

    def __init__(self, item_size, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.item_size = item_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.item_size, self.embed_size)
        self.embedding.weight = utils.truncated_normal_(self.embedding.weight, 0, 0.02)
        # stdv = np.sqrt(1. / self.item_size)
        # self.embedding.weight.data.uniform_(-stdv, stdv)

    def forward(self, itemseq_input): # inputs: [batch_size, seq_len]
        return self.embedding(itemseq_input)


class ConvDilate(nn.Module):  # causal or noncausal dilated convolution layer

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=1, causal=False):
        super(ConvDilate, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.causal = causal
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.conv.weight = utils.truncated_normal_(self.conv.weight, 0, 0.02)
        self.conv.bias.data.zero_()

    def forward(self, x):
        x_pad = self.conv_pad(x)
        output = self.conv(x_pad).squeeze(2).permute(0, 2, 1)
        return output

    def conv_pad(self, x): # pad for (non)causalCNN
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        if self.causal:
            inputs_pad = F2.pad(inputs_pad, ((self.kernel_size - 1) * self.dilation, 0))  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        else:
            inputs_pad = SamePad2d((1, self.kernel_size), self.dilation).forward(inputs_pad)
        return inputs_pad


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, causal=False):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.conv1 = ConvDilate(in_channel, out_channel, kernel_size, dilation, causal)
        self.ln1 = Layer_norm(out_channel)
        self.conv2 = ConvDilate(out_channel, out_channel, kernel_size, dilation * 2, causal)
        self.ln2 = Layer_norm(out_channel)

    def forward(self, x): # x: [batch_size, seq_len, embed_size]
        out = self.conv1(x)
        out = F2.relu(self.ln1(out))
        out = self.conv2(out)
        out = F2.relu(self.ln2(out))
        return out + x


class NextItNet_ED(nn.Module):

    def __init__(self, model_para, causal):
        super(NextItNet_ED, self).__init__()
        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']
        self.causal = causal
        rbs = [ResidualBlock(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                             dilation=dilation, causal=self.causal) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rbs)

    def forward(self, x): # inputs: [batch_size, seq_len, embed_size]
        dilate_outputs = self.residual_blocks(x)

        return dilate_outputs  # [batch_size, seq_len, dilated_channels = self.residual_channels = embed_size]


class Projector(nn.Module):

    def __init__(self, in_channel, hidden_size=64):
        super(Projector, self).__init__()
        self.conv_down = ConvDilate(in_channel, hidden_size, kernel_size=1, dilation=1)
        self.conv_up = ConvDilate(hidden_size, in_channel, kernel_size=1, dilation=1)

    def forward(self, x):
        output = self.conv_down(x)
        output = self.gelu(output)
        output = self.conv_up(output)
        return x + output

    def gelu(self, x):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.

        Returns:
          `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + torch.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
        return x * cdf


class Layer_norm(nn.Module):
    def __init__(self, channel):
        super(Layer_norm, self).__init__()
        # self.beta = torch.zeros(size, requires_grad=True)
        # self.gamma = torch.ones(size, requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel))
        # nn.init.zeros_(self.beta)
        self.gamma = nn.Parameter(torch.ones(channel))
        # nn.init.ones_(self.gamma)
        self.size = channel
        self.epsilon = 1e-8

    def forward(self, x):
        shape = x.size()
        # print(shape)
        # print(x.mean(dim=2).size())
        # print(x.std(dim=2, unbiased=False).size())
        x = (x - x.mean(dim=2).view(shape[0], shape[1], 1)) / torch.sqrt(x.var(dim=2, unbiased=False).view(shape[0], shape[1], 1) + self.epsilon)
        return self.gamma * x + self.beta


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, dilation):
        super(SamePad2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        pad_along_width = (self.kernel_size[0] - 1) * self.dilation
        pad_along_height = (self.kernel_size[1] - 1) * self.dilation
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F2.pad(input, (pad_top, pad_bottom, pad_left, pad_right))

    def __repr__(self):
        return self.__class__.__name__


