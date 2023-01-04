import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(n_embed,dim)
        self.inorm = nn.InstanceNorm1d(dim)
    def forward(self, input):
        embed = (self.embedding.weight.detach()).transpose(0,1)
        
        embed = (embed)/(torch.norm(embed,dim=0))
        #input = input / torch.norm(input, dim = 2, keepdim=True)
        flatten = input.reshape(-1, self.dim).detach()
        
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*input.shape[:-1]).detach().cpu()#.cuda()
        quantize = self.embedding(embed_ind)
        diff = (quantize - input).pow(2).mean()
        quantize_1 = input + (quantize - input).detach()
        
        return (quantize+quantize_1)/2, diff


class Decoder(nn.Module):
    def __init__(
        self, in_channel, channel
    ):
        super().__init__()
        print(in_channel)

        self.blocks = nn.Sequential(nn.Conv1d(channel*2, channel, 3, stride=1, padding=1),
                Conv1dResBlock2(channel , channel),
                Conv1dResBlock2(channel, channel),#)#,#resblocks
                nn.Upsample(in_channel))
        #self.z_scale_factors = [2,2,2]
        print(self.blocks)

    def forward(self, q_after, sp_embed, std_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]

        #std_embed = std_embed[::-1]

        x = q_after[0] + sp_embed[0]
        print(q_after[0].shape)
        print('anna')
        print(x.shape)
        x=self.blocks(x)
        print('ara',x.shape)
        #x = F.interpolate(x, scale_factor=2, mode='nearest')

        return x
    # [128, 80, 3], [1, 256, 66]


class VC_MODEL(nn.Module):
    def __init__(
        self,
        in_channel=240,
        channel=128,
        n_embed = 128,
    ):
        super().__init__()

        blocks = []

        self.enc= nn.Sequential(nn.Conv1d(in_channel, channel, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(channel,channel*2, 3, 1, 1),
                nn.Conv1d(channel*2,channel, 3, 1, 1),#resblocks
                Conv1dResBlock2(channel,channel),
                Conv1dResBlock2(channel, channel),
                #nn.ReLU(),
                nn.Conv1d(channel,channel*2, 1, 1, 1))
        print(self.enc)

        self.quantize = Quantize(channel*2, n_embed)
        self.dec = Decoder(
            in_channel ,
            channel
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, diff = self.encode(input)
        dec_1= self.decode(enc_b, sp_embed, std_block)
        idx = torch.randperm(enc_b[0].size(0))
        sp_shuffle = []
        std_shuffle = []
        for sm in (sp_embed):
            sp_shuffle += [sm[idx]]
        for std in std_block:
            std_shuffle += [std[idx]]
        
        dec_2 = self.decode(enc_b, sp_shuffle, std_shuffle)
        return dec_1, dec_2, enc_b, sp_embed, diff, idx

    def encode(self, input):
        x = input
        sp_embedding_block = []
        q_after_block = []
        std_block = []
        diff_total = 0
        print('input',x.shape)



        x = self.enc(x)
        print('output encode',x.shape)

        x_ = x - torch.mean(x, dim = 2, keepdim = True)
        std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
        std_block += [std_]
        x_ = x_ / std_

        x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)


        q_after, diff = self.quantize(x_.permute(0,2,1))
        q_after = q_after.permute(0,2,1)

        sp_embed = torch.mean(x - q_after, 2, True)
        sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

        sp_embedding_block += [sp_embed]
        q_after_block += [q_after]
        print('q',q_after_block[0].shape)
        diff_total += diff
        
        return q_after_block, sp_embedding_block, std_block, diff_total

    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class Conv1dResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)

        x = self.blocks(x)

        x += residual
        x = self.activate(x)
        return x
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels



class Conv1d(nn.Conv1d):
	"""
		Convolution 1d
		Args:
			x: (N, T, C_in)
		Returns:
			y: (N, T, C_out)
	"""

	def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                stride=1, padding='same', dilation=1, groups=1, bias=True, ln=False):

		if padding == 'same':
			padding = kernel_size // 2 * dilation
			self.even_kernel = not bool(kernel_size % 2)

		super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)

		self.activation_fn = activation_fn(inplace=True) if activation_fn is not None else None
		self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None
		self.layer_norm = nn.LayerNorm(out_channels) if ln else None

	def forward(self, x):
		y = x.transpose(1, 2)
		y = super(Conv1d, self).forward(y)
		y = y.transpose(1, 2)
		y = self.layer_norm(y) if self.layer_norm is not None else y
		y = self.activation_fn(y) if self.activation_fn is not None else y
		y = self.drop_out(y) if self.drop_out is not None else y
		y = y[:, :-1, :] if self.even_kernel else y
		return y

class Conv1dResBlock(Conv1d):
	"""
		Convolution 1d with Residual connection
		Args:
			x: (N, T, C_in)
		Returns:
			y: (N, T, C_out)
	"""
	def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                stride=1, padding='same', dilation=1, groups=1, bias=True, ln=False):

		super(Conv1dResBlock, self).__init__(in_channels, out_channels, kernel_size, activation_fn,
                                            drop_rate, stride, padding, dilation, groups=groups, bias=bias,
					    ln=ln)

	def forward(self, x):
		residual = x
		x = super(Conv1dResBlock, self).forward(x)
		x = x + residual

		return x

