from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

#根据任务类型和data选用transformerencoder
def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):
        return TSTransformerEncoder(config['cnn_embedding'],config['whether_pos_encoding'],config['cnn_kernel'],config['cnn_outchannels'],config['cnn_layer'],feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                    config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                    pos_encoding=config['pos_encoding'], activation=config['activation'],
                                    norm=config['normalization_layer'], freeze=config['freeze'],local=config['local'],window=config['window_size'])

    if (task == "classification") or (task == "regression"):
        num_labels = len(data.class_names) if (task == "classification" or task =='regression') else data.labels_df.shape[1]  # dimensionality of labels
        return TSTransformerEncoderClassiregressor(config['cnn_embedding'],config['whether_pos_encoding'],config['cnn_kernel'],config['cnn_outchannels'],config['cnn_layer'],feat_dim, max_seq_len, config['d_model'],
                                                    config['num_heads'],
                                                    config['num_layers'], config['dim_feedforward'],
                                                    num_classes=num_labels,
                                                    dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                    activation=config['activation'],
                                                    norm=config['normalization_layer'], freeze=config['freeze'],local=config['local'],window=config['window_size'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))



#选择激活函数   relu或者gelu
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class myCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dropout=0.05):
        super(myCNN, self).__init__()
        self.padding= (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, self.padding)
        #self.dropout1 = Dropout(dropout)
    def forward(self, x):
        x = self.conv1(x)
        #x = nn.functional.relu(x)
        #x = self.dropout1(x)
        return x



#positional embedding
class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




class LocalMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, *args, window_size=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) :
        seq_len = query.size(0)
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size)
            mask[i, :start] = -1e9
            mask[i, end:] = -1e9
        mask = mask.to(device=query.device, dtype=torch.float32)
        return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads,
                                               self.in_proj_weight, self.in_proj_bias,
                                               self.bias_k, self.bias_v, self.add_zero_attn,
                                               self.dropout, self.out_proj.weight, self.out_proj.bias,
                                               training=self.training,
                                               key_padding_mask=key_padding_mask,
                                               need_weights=need_weights,
                                               attn_mask=mask)





class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",local='all_atten',window=5):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.local=local
        self.window=window
        if local=='all_atten':
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        if local=='local_atten':
            self.self_attn = LocalMultiheadAttention(window_size=self.window,embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src



def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))




class TSTransformerEncoder(nn.Module):

    def __init__(self, cnn_embedding,whether_pos_encoding,cnnkernel,outputchannel_cnn,num_cnnlayer,feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='learnable', activation='gelu', norm='BatchNorm', freeze=False,local='all_atten',window=5):
        super(TSTransformerEncoder, self).__init__()

        self.local=local
        self.window=window
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads


        self.num_cnnlayer=num_cnnlayer
        self.cnn_net = nn.ModuleList()

        self.outputchannel_cnn=outputchannel_cnn
        self.outputchannel_cnn.insert(0,feat_dim)
        self.cnn_embedding=cnn_embedding
        self.whether_pos_encoding=whether_pos_encoding

        self.conv_shortcut = nn.ModuleList()
        self.dropoutcnn = Dropout(0.05)
        # self.conv_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # self.cnn =myCNN(in_channels=2, out_channels=3,kernel_size=3)
        # x = nn.functional.relu(x)
        # x = self.dropout1(x)

        if cnn_embedding:
            for i in range(num_cnnlayer):
                self.cnn_net.append(myCNN(in_channels=self.outputchannel_cnn[i], out_channels=outputchannel_cnn[i + 1],
                                          kernel_size=cnnkernel))
                if self.outputchannel_cnn[i] != outputchannel_cnn[i + 1]:
                    self.conv_shortcut.append(
                        nn.Conv1d(self.outputchannel_cnn[i], out_channels=outputchannel_cnn[i + 1], kernel_size=1))

            self.project_inp = nn.Linear(outputchannel_cnn[-1], d_model)
            if whether_pos_encoding:
                self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)



        else:
            self.project_inp = nn.Linear(feat_dim, d_model)
            self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)


        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation,local=self.local,window=self.window)


        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        a = 0
        if self.cnn_embedding:
            inp1 = X.permute(0, 2, 1)
            for i in range(self.num_cnnlayer):
                op1 = self.cnn_net[i](inp1)
                if op1.size() != inp1.size():
                    res = self.conv_shortcut[a](inp1)
                    a = a + 1
                else:
                    res = inp1
                op1 = op1 + res
                inp1 = nn.functional.relu(op1)
                inp1 = self.dropoutcnn(inp1)

            X = inp1.permute(0, 2, 1)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        if self.whether_pos_encoding:
            inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, cnn_embedding,whether_pos_encoding,cnnkernel,outputchannel_cnn,num_cnnlayer,feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='learnable', activation='gelu', norm='BatchNorm', freeze=False,local='all_atten',window=5):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.local=local
        self.window=window
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.num_cnnlayer=num_cnnlayer
        self.cnn_net = nn.ModuleList()

        self.outputchannel_cnn=outputchannel_cnn
        self.outputchannel_cnn.insert(0,feat_dim)
        self.cnn_embedding=cnn_embedding
        self.whether_pos_encoding=whether_pos_encoding

        self.conv_shortcut = nn.ModuleList()
        self.dropoutcnn = Dropout(0.05)


        if cnn_embedding:
            for i in range(num_cnnlayer ):
                self.cnn_net.append(myCNN(in_channels=self.outputchannel_cnn[i], out_channels=outputchannel_cnn[i + 1],kernel_size=cnnkernel))
                if self.outputchannel_cnn[i]!=outputchannel_cnn[i + 1]:
                    self.conv_shortcut.append(nn.Conv1d(self.outputchannel_cnn[i],out_channels=outputchannel_cnn[i + 1], kernel_size=1))


            self.project_inp = nn.Linear(outputchannel_cnn[-1], d_model)
            if whether_pos_encoding:
                self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)



        else:
            self.project_inp = nn.Linear(feat_dim, d_model)
            self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation,local=self.local,window=self.window)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes


        #用作多任务学习
        self.output_layer_a = self.build_output_module(d_model, 2)


    def build_output_module(self, d_model, num_classes):
        output_layer = nn.Linear(d_model, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        a=0
        if self.cnn_embedding:
            inp1 = X.permute(0, 2, 1)
            for i in range(self.num_cnnlayer):
                op1=self.cnn_net[i](inp1)
                if op1.size() != inp1.size():
                    res=self.conv_shortcut[a](inp1)
                    a=a+1
                else:
                    res=inp1
                op1=op1+res
                inp1 = nn.functional.relu(op1)
                inp1 = self.dropoutcnn(inp1)


            X=inp1.permute(0, 2, 1)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        if self.whether_pos_encoding:
            inp1 = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp1, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings


        #output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output_a=self.output_layer_a(output)  # (batch_size, num_classes)


        inp = inp.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        inp = inp * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        inp = inp.reshape(inp.shape[0], -1)  # (batch_size, seq_length * d_model)


        return output_a,output,inp

        #return output_m, output_a, output






