import math
from math import sqrt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# from mamba_ssm import Mamba
#from torchvision.models.vision_transformer import MLPBlock
# from .model import SelfAttention
# from .model_wavenet import AnDi_Wave


class AnDi_Wave(nn.Module):
    def __init__(self, input_dim, filters, kernel_size, dilation_depth):
        super().__init__()
        self.dilation_depth =  dilation_depth
        
        self.dilations = [2**i for i in range(dilation_depth)]
        self.conv1d_tanh = nn.ModuleList([nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, 
                                                    padding=int((dilation*(kernel_size-1))/2), dilation=dilation) 
                                                    for dilation in self.dilations])
        self.conv1d_sigm = nn.ModuleList([nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, 
                                                    padding=int((dilation*(kernel_size-1))/2), dilation=dilation) 
                                                    for dilation in self.dilations]) 
        self.conv1d_0 = nn.Conv1d(in_channels=input_dim, out_channels=filters, 
                                  kernel_size=kernel_size, padding=1)
        self.conv1d_1 = nn.Conv1d(in_channels=filters, out_channels=filters, 
                                  kernel_size=1, padding=0)
        self.post = nn.Sequential(nn.BatchNorm1d(filters), nn.Dropout(0.1))
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        # WaveNet Block
        x = self.conv1d_0(x)
        res_x = x
        
        for i in range(self.dilation_depth):
            tahn_out = torch.tanh(self.conv1d_tanh[i](x))
            sigm_out = torch.sigmoid(self.conv1d_sigm[i](x))
            x = tahn_out * sigm_out
            x = self.conv1d_1(x)
            res_x = res_x + x
        
        x = self.post(res_x)
        out = self.pool(x)
        
        return out, x


class TrajModule_ConvLSTM(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256, num_head=8, dropout=0.5, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.fc0 = nn.Linear(embed_dim, hidden_dim)
        self.res = AnDi_Wave(embed_dim, filters=hidden_dim, kernel_size=3, dilation_depth=5)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=dropout, bidirectional=True)

        self.dropout = nn.Dropout(p=dropout)
        # self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.attentions = nn.ModuleList([
        #     SelfAttention(hidden_dim * 2, hidden_dim * 2, hidden_dim * 2) for _ in range(self.num_head)
        #     ])
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.mlp1 = nn.Linear(hidden_dim * 2 * self.num_head, hidden_dim)
        # self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.mlp1 = nn.Sequential(MLPBlock(hidden_dim * self.num_head, mlp_dim, dropout), nn.Linear(hidden_dim * self.num_head, hidden_dim))
        # self.mlp2 = nn.Sequential(MLPBlock(hidden_dim * 2, mlp_dim, dropout), nn.Linear(hidden_dim * 2, hidden_dim))
    
    def forward(self, x):
        x_attn = F.relu(self.fc0(x))
        _, x_conv = self.res(x.permute(0, 2, 1))
        x_conv = x_conv.permute(0, 2, 1) + x_attn
        # x_attn = x_attn_res

        output_lstm, (h, c) = self.lstm(x_attn)
        output_lstm = F.relu(self.mlp2(self.dropout(output_lstm))) + x_attn
        # output_lstm = self.dropout(self.mlp2(output_lstm)) + x_attn

        output = torch.cat((x_conv, output_lstm), dim=-1)
        output = self.mlp_out(output)

        return output

class TrajModule_ConvLSTM_seq(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256, num_head=8, dropout=0.5, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.fc0 = nn.Linear(embed_dim, hidden_dim)
        self.res = AnDi_Wave(embed_dim, filters=hidden_dim, kernel_size=3, dilation_depth=5)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, dropout=dropout, bidirectional=True)

        self.dropout = nn.Dropout(p=dropout)
        # self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.attentions = nn.ModuleList([
        #     SelfAttention(hidden_dim * 2, hidden_dim * 2, hidden_dim * 2) for _ in range(self.num_head)
        #     ])
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim * 2 * self.num_head, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.mlp1 = nn.Sequential(MLPBlock(hidden_dim * self.num_head, mlp_dim, dropout), nn.Linear(hidden_dim * self.num_head, hidden_dim))
        # self.mlp2 = nn.Sequential(MLPBlock(hidden_dim * 2, mlp_dim, dropout), nn.Linear(hidden_dim * 2, hidden_dim))
    
    def forward(self, x):
        x_attn = self.fc0(x)
        # x_attn = x_attn
        # x_attn_res = x_attn
        _, x_conv = self.res(x.permute(0, 2, 1))
        x_conv = x_conv.permute(0, 2, 1)
        # x_attn = x_attn_res

        output_lstm, (h, c) = self.lstm(x_conv)
        # output_lstm = self.dropout(output_lstm)
        output = self.dropout(self.mlp2(output_lstm)) + x_attn

        return output 

class ConvLSTM_enc_dec(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder = nn.ModuleList([
            TrajModule_ConvLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder.append(TrajModule_ConvLSTM(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_motion = copy.deepcopy(self.encoder)
        self.decoder_alpha = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_alpha.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        self.decoder_D = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_D.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_state = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_state.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_state.append(nn.Linear(hidden_dim, 4))
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        in_x = F.relu(self.in_proj(x))
        motion = torch.zeros_like(in_x)
        motion[:, 1:, :] = in_x[:, 1:, :] - in_x[:, :-1, :]
        out_enc = in_x
        for layer in self.encoder:
            out_enc = layer(out_enc)
        out_motion = motion
        for layer in self.encoder_motion:
            out_motion = layer(out_motion)
        out_enc = out_enc + out_motion
        out_alpha = out_enc
        out_D = out_enc
        out_state = out_enc
        for layer in self.decoder_alpha:
            out_alpha = layer(out_alpha)
        for layer in self.decoder_D:
            out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        for layer in self.decoder_state:
            out_state = layer(out_state)
        return {'alpha': out_alpha, 'D': F.leaky_relu(out_D), 'state': out_state}#, 'CP': out_CP,
    

class ConvLSTM_enc_dec_msd(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder_motion = nn.ModuleList([
            TrajModule_ConvLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder_motion.append(TrajModule_ConvLSTM(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.encoder_motion = copy.deepcopy(self.encoder)
        self.decoder_alpha = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_alpha.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        self.decoder_D = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_D.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_state = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_state.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_state.append(nn.Linear(hidden_dim, 4))
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        # in_x = F.relu(self.in_proj(x))
        motion = torch.zeros_like(x)
        motion[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        motion = F.relu(self.in_proj(motion))
        # for layer in self.encoder:
        #     out_enc = layer(out_enc)
        out_motion = motion
        for layer in self.encoder_motion:
            out_motion = layer(out_motion)
        # out_enc = out_enc + out_motion
        out_enc = out_motion
        out_alpha = out_enc
        out_D = out_enc
        out_state = out_enc
        for layer in self.decoder_alpha:
            out_alpha = layer(out_alpha)
        for layer in self.decoder_D:
            out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        for layer in self.decoder_state:
            out_state = layer(out_state)
        return {'alpha': out_alpha, 'D': F.leaky_relu(out_D), 'state': out_state}#, 'CP': out_CP,

class ConvLSTM_enc_dec_motion(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.in_motion_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder = nn.ModuleList([
            TrajModule_ConvLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder.append(TrajModule_ConvLSTM(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_motion = copy.deepcopy(self.encoder)
        self.decoder_alpha = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_alpha.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1] * 2, hidden_dim_list[0], num_head, dropout))
        self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        self.decoder_D = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_D.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1] * 2, hidden_dim_list[0], num_head, dropout))
        self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_state = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_state.append(TrajModule_ConvLSTM(hidden_dim_list[layer_num - 1] * 2, hidden_dim_list[0], num_head, dropout))
        self.decoder_state.append(nn.Linear(hidden_dim, 5))
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        in_x = F.relu(self.in_proj(x))
        motion = torch.zeros_like(x)
        motion[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        in_motion = F.relu(self.in_motion_proj(motion))
        out_enc = in_x
        # out_enc = in_motion
        for layer in self.encoder:
            out_enc = layer(out_enc)
        out_motion = in_motion
        for layer in self.encoder_motion:
            out_motion = layer(out_motion)
        out_enc = torch.cat((out_enc, out_motion), dim=-1)
        # out_enc = torch.mul(out_enc, out_motion)
        out_alpha = out_enc
        out_D = out_enc
        out_state = out_enc
        for layer in self.decoder_alpha:
            out_alpha = layer(out_alpha)
        for layer in self.decoder_D:
            out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        for layer in self.decoder_state:
            out_state = layer(out_state)
        return {'alpha': out_alpha, 'D': F.leaky_relu(out_D), 'state': out_state}#, 'CP': out_CP,


class Module_Conv(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256, num_head=8, dropout=0.5, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.fc0 = nn.Linear(embed_dim, hidden_dim)
        self.res = AnDi_Wave(embed_dim, filters=hidden_dim, kernel_size=3, dilation_depth=5)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        x_attn = F.relu(self.fc0(x))
        _, x_conv = self.res(x.permute(0, 2, 1))
        x_conv = x_conv.permute(0, 2, 1) + x_attn
        return x_conv
        # x_attn = x_attn_res

        output_lstm, (h, c) = self.lstm(x_attn)
        output_lstm = F.relu(self.mlp2(self.dropout(output_lstm))) + x_attn
        # output_lstm = self.dropout(self.mlp2(output_lstm)) + x_attn

        output = torch.cat((x_conv, output_lstm), dim=-1)
        output = self.mlp_out(output)

        return output


class ConvNet_motion(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.in_motion_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder = nn.ModuleList([
            # Module_Conv(embed_dim, hidden_dim, num_head, dropout)
            TrajModule_ConvLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder.append(Module_Conv(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_motion = copy.deepcopy(self.encoder)
        self.decoder_alpha = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_alpha.append(Module_Conv(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        self.decoder_D = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_D.append(Module_Conv(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_state = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_state.append(Module_Conv(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_state.append(nn.Linear(hidden_dim, 5)) # leave out 1 virtual class (maybe performs better)
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        in_x = F.relu(self.in_proj(x))
        motion = torch.zeros_like(x)
        motion[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        in_motion = F.relu(self.in_motion_proj(motion))
        out_enc = in_x
        # out_enc = in_motion
        for layer in self.encoder:
            out_enc = layer(out_enc)
        out_motion = in_motion
        for layer in self.encoder_motion:
            out_motion = layer(out_motion)
        # out_enc = torch.cat((out_enc, out_motion), dim=-1)
        out_enc = torch.mul(out_enc, out_motion)
        out_alpha = out_enc
        out_D = out_enc
        out_state = out_enc
        for layer in self.decoder_alpha:
            out_alpha = layer(out_alpha)
        for layer in self.decoder_D:
            out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        for layer in self.decoder_state:
            out_state = layer(out_state)
        return {'alpha': out_alpha, 'D': F.leaky_relu(out_D), 'state': out_state}#, 'CP': out_CP,


class ConvNet_motion_changepoint(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_coord_proj = nn.Linear(input_dim, embed_dim)
        self.in_param_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder_coord = nn.ModuleList([
            # Module_Conv(embed_dim, hidden_dim, num_head, dropout)
            TrajModule_ConvLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder_coord.append(Module_Conv(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_param = copy.deepcopy(self.encoder_coord)

        self.decoder_cp = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_cp.append(Module_Conv(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_cp.append(nn.Linear(hidden_dim, 3)) # leave out 1 virtual class (maybe performs better)
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x, param):
        in_x = F.relu(self.in_coord_proj(x))
        in_param = F.relu(self.in_param_proj(param))
        out_enc = in_x
        # out_enc = in_motion
        for layer in self.encoder_coord:
            out_enc = layer(out_enc)
        out_motion = in_param
        for layer in self.encoder_param:
            out_motion = layer(out_motion)
        # out_enc = torch.cat((out_enc, out_motion), dim=-1)
        out_enc = torch.mul(out_enc, out_motion)
        out_cp = out_enc
        for layer in self.decoder_cp:
            out_cp = layer(out_cp)
        return out_cp


class TrajModule_GatedLSTM(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256, num_head=8, dropout=0.5, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.fc0 = nn.Linear(embed_dim, hidden_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # self.res = AnDi_Wave(embed_dim, filters=hidden_dim, kernel_size=3, dilation_depth=5)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=dropout, bidirectional=True)

        self.dropout = nn.Dropout(p=dropout)
        # self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.attentions = nn.ModuleList([
        #     SelfAttention(hidden_dim * 2, hidden_dim * 2, hidden_dim * 2) for _ in range(self.num_head)
        #     ])
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.mlp1 = nn.Linear(hidden_dim * 2 * self.num_head, hidden_dim)
        # self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, hidden_dim)
        # self.mlp1 = nn.Sequential(MLPBlock(hidden_dim * self.num_head, mlp_dim, dropout), nn.Linear(hidden_dim * self.num_head, hidden_dim))
        # self.mlp2 = nn.Sequential(MLPBlock(hidden_dim * 2, mlp_dim, dropout), nn.Linear(hidden_dim * 2, hidden_dim))
    
    def forward(self, x):
        x_attn = F.leaky_relu(self.fc0(x))
        x_res = F.leaky_relu(self.fc1(x))

        output_lstm, (h, c) = self.lstm(x_attn)
        output_lstm = F.leaky_relu(self.mlp2(self.dropout(output_lstm)))
        # output_lstm = self.dropout(self.mlp2(output_lstm)) + x_attn

        output = x_res * output_lstm
        output = self.mlp_out(output)

        return output


class GatedLSTM(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.max_len = 200
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.motion_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder = nn.ModuleList([
            TrajModule_GatedLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder.append(TrajModule_GatedLSTM(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_motion = copy.deepcopy(self.encoder)
        self.decoder_alpha = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_alpha.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        self.decoder_D = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_D.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_state = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_state.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_state.append(nn.Linear(hidden_dim, 4))
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x, padding_masks):
        in_x = F.leaky_relu(self.in_proj(x))
        motion = torch.zeros_like(in_x)
        motion[:, 1:, :] = in_x[:, 1:, :] - in_x[:, :-1, :]
        motion = torch.zeros_like(x)
        motion[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        motion = self.motion_proj(motion)
        out_enc = in_x
        for layer in self.encoder:
            out_enc = layer(out_enc)
        out_motion = motion
        for layer in self.encoder_motion:
            out_motion = layer(out_motion)
        out_enc = out_enc * out_motion
        out_alpha = out_enc
        out_D = out_enc
        out_state = out_enc
        for layer in self.decoder_alpha:
            out_alpha = layer(out_alpha)
        for layer in self.decoder_D:
            out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        for layer in self.decoder_state:
            out_state = layer(out_state)
        # return {'alpha': out_alpha, 'D': F.leaky_relu(out_D), 'state': out_state}#, 'CP': out_CP,
        out = torch.cat((out_alpha, out_D, out_state), dim=-1)
        out = out * padding_masks.unsqueeze(-1)
        return out, x, out

class GatedLSTM_msd(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.motion_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder = nn.ModuleList([
            TrajModule_GatedLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder.append(TrajModule_GatedLSTM(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_motion = copy.deepcopy(self.encoder)
        self.decoder_alpha = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_alpha.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        self.decoder_D = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_D.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_state = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_state.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_state.append(nn.Linear(hidden_dim, 4))
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        # in_x = F.relu(self.in_proj(x))
        # motion = torch.zeros_like(in_x)
        # motion[:, 1:, :] = in_x[:, 1:, :] - in_x[:, :-1, :]
        motion = torch.zeros_like(x)
        motion[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        motion = self.motion_proj(motion)
        # out_enc = in_x
        # for layer in self.encoder:
        #     out_enc = layer(out_enc)
        out_motion = motion
        for layer in self.encoder_motion:
            out_motion = layer(out_motion)
        # out_enc = out_enc * out_motion
        out_enc = out_motion
        out_alpha = out_enc
        out_D = out_enc
        out_state = out_enc
        for layer in self.decoder_alpha:
            out_alpha = layer(out_alpha)
        for layer in self.decoder_D:
            out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        for layer in self.decoder_state:
            out_state = layer(out_state)
        return {'alpha': out_alpha, 'D': F.leaky_relu(out_D), 'state': out_state}#, 'CP': out_CP,


class GatedLSTM_changepoint(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.motion_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder = nn.ModuleList([
            TrajModule_GatedLSTM(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder.append(TrajModule_GatedLSTM(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_motion = copy.deepcopy(self.encoder)
        # self.decoder_alpha = nn.ModuleList([])
        # # for j in range(layer_num - 1, 1, -1):
        # self.decoder_alpha.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        # self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        # self.decoder_D = nn.ModuleList([])
        # # for j in range(layer_num - 1, 1, -1):
        # self.decoder_D.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        # self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_cp = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_cp.append(TrajModule_GatedLSTM(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_cp.append(nn.Linear(hidden_dim, 2))
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        in_x = F.relu(self.in_proj(x))
        # motion = torch.zeros_like(in_x)
        # motion[:, 1:, :] = in_x[:, 1:, :] - in_x[:, :-1, :]
        # motion = torch.zeros_like(x)
        # motion[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        # motion = self.motion_proj(motion)
        out_enc = in_x
        for layer in self.encoder:
            out_enc = layer(out_enc)
        # out_motion = motion
        # for layer in self.encoder_motion:
        #     out_motion = layer(out_motion)
        # out_enc = out_enc * out_motion
        # out_alpha = out_enc
        # out_D = out_enc
        # out_state = out_enc
        # for layer in self.decoder_alpha:
        #     out_alpha = layer(out_alpha)
        # for layer in self.decoder_D:
        #     out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        out_cp = out_enc
        for layer in self.decoder_cp:
            out_cp = layer(out_cp)
        return out_cp


class TrajModule_LinearMamba(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256, num_head=8, dropout=0.5, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.fc0 = nn.Linear(embed_dim, hidden_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.res = AnDi_Wave(embed_dim, filters=hidden_dim, kernel_size=3, dilation_depth=5)
        # self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=dropout, bidirectional=True)
        # self.mamba1 = Mamba(d_model=hidden_dim)
        # self.mamba2 = Mamba(d_model=200)

        self.dropout = nn.Dropout(p=dropout)
        # self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)
        # self.dropout2 = nn.Dropout(p=0.2)
        self.attentions = nn.ModuleList([
            SelfAttention(hidden_dim, hidden_dim, hidden_dim) for _ in range(self.num_head)
            ])
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim * self.num_head, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.mlp1 = nn.Sequential(MLPBlock(hidden_dim * self.num_head, mlp_dim, dropout), nn.Linear(hidden_dim * self.num_head, hidden_dim))
        # self.mlp2 = nn.Sequential(MLPBlock(hidden_dim * 2, mlp_dim, dropout), nn.Linear(hidden_dim * 2, hidden_dim))
    
    def forward(self, x):
        x_attn = F.relu(self.fc0(x))
        _, x_res = self.res(x.permute(0, 2, 1))
        x_res = x_res.permute(0, 2, 1)

        output_list = []
        for attention in self.attentions:
            output_list.append(attention(x_attn))
        output_attn = torch.cat(output_list, dim=-1)
        output_attn = self.dropout(self.mlp1(output_attn))

        # output_lstm, (h, c) = self.lstm(output_attn)
        # output_mamba1 = self.mamba1(output_attn)
        # output_mamba2 = self.mamba2(output_attn.permute(0, 2, 1)).permute(0, 2, 1)
        # output_mamba = torch.cat((output_mamba1, output_mamba2), dim=-1)
        # output_mamba = self.dropout(self.mlp2(output_mamba))

        output = x_res + output_attn
        # output = self.mlp2(output)
        return output
    
class GatedMamba(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=256, num_head=8, layer_num=1, dropout=0.1, mlp_dim=3072) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        hidden_dim_list = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, hidden_dim * 2]
        self.encoder = nn.ModuleList([
            TrajModule_LinearMamba(embed_dim, hidden_dim, num_head, dropout)
            # TrajModule_AttnMamba(embed_dim, hidden_dim, num_head, dropout), # Modified
        ])
        for i in range(layer_num - 1):
            # self.layers_alpha.append(TrajModule_AttnMamba(hidden_dim, hidden_dim, num_head, dropout))
            self.encoder.append(TrajModule_LinearMamba(hidden_dim_list[i], hidden_dim_list[i + 1], num_head, dropout))
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_motion = copy.deepcopy(self.encoder)
        self.decoder_alpha = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_alpha.append(TrajModule_LinearMamba(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_alpha.append(nn.Linear(hidden_dim, 1))

        self.decoder_D = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_D.append(TrajModule_LinearMamba(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_D.append(nn.Linear(hidden_dim, 1))

        self.decoder_state = nn.ModuleList([])
        # for j in range(layer_num - 1, 1, -1):
        self.decoder_state.append(TrajModule_LinearMamba(hidden_dim_list[layer_num - 1], hidden_dim_list[0], num_head, dropout))
        self.decoder_state.append(nn.Linear(hidden_dim, 4))
        # self.head_alpha = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))
        # self.head_D = nn.Sequential(MLPBlock(hidden_dim, mlp_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        in_x = F.relu(self.in_proj(x))
        motion = torch.zeros_like(in_x)
        motion[:, 1:, :] = in_x[:, 1:, :] - in_x[:, :-1, :]
        out_enc = in_x
        for layer in self.encoder:
            out_enc = layer(out_enc)
        out_motion = motion
        for layer in self.encoder_motion:
            out_motion = layer(out_motion)
        out_enc = out_enc + out_motion
        out_alpha = out_enc
        out_D = out_enc
        out_state = out_enc
        for layer in self.decoder_alpha:
            out_alpha = layer(out_alpha)
        for layer in self.decoder_D:
            out_D = layer(out_D)
        # out_CP = F.sigmoid(self.head_CP(x_all))
        for layer in self.decoder_state:
            out_state = layer(out_state)
        return {'alpha': out_alpha, 'D': F.leaky_relu(out_D), 'state': out_state}#, 'CP': out_CP,
