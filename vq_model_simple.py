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

        blocks_refine = []
        resblock = []
        #num_groups = 4
        
        #self.block0 = GBlock(in_channel//8, in_channel//8, channel, num_groups)
        # L2-norm of input vector into 1 on every step
        #self.norm = torch.norm(in_channel, dim= 1, keepdim = True)

        blocks = []
        blocks += [
            nn.Sequential(*[
                nn.Conv1d(in_channel // 2, channel, 3, stride=1, padding=1),
            ])]

        for i in range(2):
            print(channel)
            block = [ResidualBlock(channel,channel)]#(in_channel//2**(i+1), in_channel//2**(i))] #de128 a 128
            print(block)
            resblock += block
        blocks_refine += [nn.Upsample(channel, 2)]


        self.blocks = nn.ModuleList(blocks)
        print(self.blocks)
        self.blocks_refine = nn.ModuleList(blocks_refine[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])
        self.z_scale_factors = [2,2,2]

    def forward(self, q_after, sp_embed, std_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]
        std_embed = std_embed[::-1]
        x = 0

        for i, (block,  block_refine, res,scale_factor) in enumerate(zip(self.blocks,self.blocks_refine, self.resblock,self.z_scale_factors)):

            x = x + res(q_after[i] + sp_embed[i])

            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')

            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
        return x


class VC_MODEL(nn.Module):
    def __init__(
        self,
        in_channel=240,
        channel=128,
        n_embed = 128,
    ):
        super().__init__()

        blocks = []
        for i in range(1):
            blocks += [
            nn.Sequential(*[
                nn.Conv1d(in_channel//2**(i), channel, 4, stride=2, padding=1),
                nn.ReLU(), #change
                nn.Conv1d(channel, in_channel//2**(i+1), 3, 1, 1),
                
            ])]

        # ResBlock x2 size 128
        for i in range(2):
            blocks += [ResidualBlock(in_channel//2**(i+1),channel)]#in_channel//2**(i+1), channel)]

        blocks += [
            nn.Sequential(*[
                nn.ReLU(),  # change
                nn.Conv1d(channel, in_channel // 2 ** (i + 1), 1, 1, 1),
            ])]
        self.enc = nn.ModuleList(blocks)
        
        quantize_blocks = []
        
        for i in range(1):
            quantize_blocks += [
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        self.quantize = nn.ModuleList(quantize_blocks)

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




        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)

            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
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

class ResidualBlock(nn.Module):
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

