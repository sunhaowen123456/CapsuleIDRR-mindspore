import os
import pickle
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Uniform
from mindnlp.transformers import BertModel, BertConfig
# from transformers import BertModel, BertTokenizer, BertConfig
import CatCapsuleModels


class Atten(nn.Cell):
    def __init__(self, in_dim, conf):
        super(Atten, self).__init__()
        self.conf = conf
        self.w = nn.Conv1d(in_dim, in_dim, 1, pad_mode='valid', has_bias=True)
        self.temper = np.power(in_dim, 0.5)
        self.dropout = nn.Dropout(p=self.conf.attn_dropout)
        self.softmax = nn.Softmax(-1)

    def construct(self, q, v):
        q_ = self.w(q.swapaxes(1, 2)).swapaxes(1, 2)
        attn = ops.bmm(q_, v.swapaxes(1, 2)) / self.temper
        vr = ops.bmm(self.dropout(self.softmax(attn)), v)
        qr = ops.bmm(self.dropout(self.softmax(attn.swapaxes(1, 2))), q)
        vr = ops.topk(vr, k=self.conf.attn_topk, dim=1)[0]
        vr = vr.view(vr.shape[0], -1)
        qr = ops.topk(qr, k=self.conf.attn_topk, dim=1)[0]
        qr = qr.view(qr.shape[0], -1)
        return qr, vr, attn


class AttenCapsule(nn.Cell):
    def __init__(self, in_dim, conf):
        super(AttenCapsule, self).__init__()
        self.conf = conf
        self.w = nn.Conv1d(in_dim, in_dim, 1, pad_mode='valid', has_bias=True)
        self.temper = np.power(in_dim, 0.5)
        self.dropout = nn.Dropout(p=self.conf.attn_dropout)
        self.softmax = nn.Softmax(-1)

    def construct(self, q, v):
        q_ = self.w(q.swapaxes(1, 2)).swapaxes(1, 2)
        attn = ops.bmm(q_, v.swapaxes(1, 2)) / self.temper   # [128,50,50]
        vr = ops.bmm(self.dropout(self.softmax(attn)), v)
        qr = ops.bmm(self.dropout(self.softmax(attn.swapaxes(1, 2))), q)
        vr = ops.topk(vr, k=self.conf.capsule_att_top, dim=1)[0]  # capsule_att_top = 18
        qr = ops.topk(qr, k=self.conf.capsule_att_top, dim=1)[0]  # [128,18,400]
        return qr, vr, attn


class Highway(nn.Cell):
    def __init__(self, size):
        super(Highway, self).__init__()
        self.highway_linear = nn.Dense(size, size)
        self.gate_linear = nn.Dense(size, size)
        self.nonlinear = nn.ReLU()

    def construct(self, input):
        gate = ops.sigmoid(self.gate_linear(input))
        m = self.nonlinear(self.highway_linear(input))
        return gate * m + (1 - gate) * input


class CNNLayer(nn.Cell):  # 一层一维卷积，输出的shape不变
    def __init__(self, conf, in_dim, k, res=True):
        super(CNNLayer, self).__init__()
        self.conf = conf
        self.res = res  # 是否使用残差连接
        self.conv = nn.Conv1d(in_dim, in_dim * 2, k, stride=1, padding=k // 2, pad_mode='pad')  #has_bias=True, weight_init='Uniform', bias_init="Zero" 输入输出长度相同 维度不同  weight_init='Uniform'
        self.dropout = nn.Dropout(p=self.conf.cnn_dropout)

    def construct(self, input):
        output = self.dropout(input.swapaxes(1, 2))
        tmp = self.conv(output)  # dim翻倍
        if tmp.shape[2] > output.shape[2]:
            output = tmp[:, :, 1:]
        else:
            output = tmp
        output = output.swapaxes(1, 2)
        a, b = ops.chunk(output, 2, axis=2)
        output = a * ops.sigmoid(b)  # 每个元素sigmoid  门控线性单元GLU
        if self.res:
            output = output + input
        return output


class CharLayer(nn.Cell):
    def __init__(self, char_table, conf):
        super(CharLayer, self).__init__()
        self.conf = conf
        lookup, length = char_table
        self.char_embed = nn.Embedding(vocab_size=self.conf.char_num, embedding_size=self.conf.char_embed_dim,
                                       padding_idx=self.conf.char_padding_idx)
        self.lookup = nn.Embedding(lookup.shape[0], lookup.shape[1])
        #ms.Tensor(lookup, dtype=ms.float32)
        self.lookup.embedding_table.set_data(ms.Tensor(lookup, dtype=ms.float32))  # 复制预训练的字符级嵌入
        self.lookup.embedding_table.requires_grad = False
        # self.lookup.weight.data.copy_(lookup)
        # self.lookup.weight.requires_grad = False
        self.convs = nn.CellList()  # 2层卷积层 仅卷积核大小不同
        for i in range(self.conf.char_filter_num):
            self.convs.append(nn.Conv1d(
                self.conf.char_embed_dim, self.conf.char_enc_dim, self.conf.char_filter_dim[i],
                stride=1, padding=self.conf.char_filter_dim[i] // 2, pad_mode='pad', has_bias=True,
                weight_init="XavierUniform"
            ))
        self.nonlinear = nn.Tanh()  # self.char_hid_dim = self.char_enc_dim * self.char_filter_num = 50*2
        self.mask = nn.Embedding(lookup.shape[0], self.conf.char_hid_dim)  # mask前两行
        x = np.ones((self.mask.embedding_table.shape[0], self.mask.embedding_table.shape[1]))
        x[0:2, :] = 0
        #ms.Tensor(lookup, dtype=ms.float32)
        self.mask.embedding_table.set_data(ms.Tensor(lookup, dtype=ms.float32))
        # self.mask.weight.data.fill_(1)
        # self.mask.weight.data[0].fill_(0)
        # self.mask.weight.data[1].fill_(0)
        self.mask.embedding_table.requires_grad = False
        self.highway = Highway(self.conf.char_hid_dim)
        del x
        del lookup
        del length

    def construct(self, input):
        charseq = ms.Tensor(self.lookup(input), ms.int64).view(input.shape[0] * input.shape[1], -1)
        charseq = self.char_embed(charseq).swapaxes(1, 2)
        conv_out = []
        for i in range(self.conf.char_filter_num):
            tmp = self.nonlinear(self.convs[i](charseq))
            if tmp.shape[2] > charseq.shape[2]:
                tmp = tmp[:, :, 1:]
            tmp = ops.topk(tmp, k=1)[0]  # 默认最后一个维度
            conv_out.append(ops.squeeze(tmp, axis=2))  # 删去最后一个维度再append
        hid = ops.cat(conv_out, axis=1)
        hid = self.highway(hid)
        hid = hid.view(input.shape[0], input.shape[1], -1)  # 形状调整为(batch_size, sequence_length, char_hid_dim)
        mask = self.mask(input)
        hid = hid * mask
        return hid


# class ElmoLayer(nn.Module):
#     #https://blog.csdn.net/CSTGYinZong/article/details/121833353
#     def __init__(self, char_table, conf):
#         super(ElmoLayer, self).__init__()
#         self.conf = conf
#         #lookup:torch.Size([32262, 50])   lenth:torch.Size([32262, 1])
#         lookup, length = char_table
#         self.lookup = nn.Embedding(lookup.size(0), lookup.size(1))
#         self.lookup.weight.data.copy_(lookup)
#         self.lookup.weight.requires_grad = False
#         #os.path.expanduser 用于将路径字符串中的波浪线（~）扩展为用户的主目录
#         # self.elmo = Elmo(
#         #     os.path.expanduser(self.conf.elmo_options), os.path.expanduser(self.conf.elmo_weights),
#         #     num_output_representations=2, do_layer_norm=False, dropout=self.conf.embed_dropout
#         # )
#         # for p in self.elmo.parameters():
#         #     p.requires_grad = False
#         self.w = nn.Parameter(torch.Tensor([0.5, 0.5]))
#         self.gamma = nn.Parameter(torch.ones(1))
#         self.conv = nn.Conv1d(1024, self.conf.elmo_dim, 1)
#         #将卷积核的数值均匀初始化 Xavier的作用就是预防一些参数过大或过小的情况
#         nn.init.xavier_uniform_(self.conv.weight)
#         self.conv.bias.data.fill_(0)
#
#     def forward(self, input):
#         charseq = self.lookup(input).long()
#         res = self.elmo(charseq)['elmo_representations']
#         w = F.softmax(self.w, dim=0)
#         res = self.gamma * (w[0] * res[0] + w[1] * res[1])
#         res = self.conv(res.transpose(1, 2)).transpose(1, 2)
#         return res


# RNN + CNN + CatCapsule
class CatCapsuleEncoder(nn.Cell):
    def __init__(self, conf, we_tensor, char_table=None, sub_table=None, use_cuda=False, attnvis=False):
        super(CatCapsuleEncoder, self).__init__()
        self.conf = conf
        self.attnvis = attnvis
        # if self.conf.use_rnn:
        #     self.usecuda = use_cuda
        #we使用的是./data/processed/ji/we.pkl
        # self.embed = nn.Embedding(we_tensor.size(0), we_tensor.size(1))
        # self.embed.weight.data.copy_(we_tensor)
        # self.embed.weight.requires_grad = False
        self.embed = nn.Embedding(we_tensor.shape[0], we_tensor.shape[1])
        #self.embed.embedding_table.set_data(ms.Tensor(we_tensor, dtype=ms.float32))
        self.embed.weight.set_data(ms.Tensor(we_tensor, dtype=ms.float32))
        #self.embed.embedding_table.requires_grad = False
        self.embed.weight.requires_grad = False
        #need_char=need_sub=False
        if self.conf.need_char:
            self.charenc = CharLayer(char_table, self.conf)
        if self.conf.need_sub:
            self.charenc = CharLayer(sub_table, self.conf)
        # need_elmo=True
        # if self.conf.need_elmo:
        #     self.elmo = ElmoLayer(char_table, self.conf)
        if self.conf.need_bert:
            self.bert = BertLayer()
        self.dropout = nn.Dropout(p=self.conf.embed_dropout)

        # self.block1 = nn.ModuleList()
        # self.block2 = nn.ModuleList()
        self.blocks = nn.CellList()
        self.attn = Atten(self.conf.cnn_dim, self.conf)
        # block1和block2 分别代表了2个句子CNN卷积操作
        for i in range(self.conf.cnn_layer_num * 2):
            self.blocks.append(CNNLayer(self.conf, self.conf.cnn_dim, self.conf.cnn_kernal_size[i // 2]))
            # self.block1.append(CNNLayer(self.conf, self.conf.cnn_dim, self.conf.cnn_kernal_size[i]))
            # self.block2.append(CNNLayer(self.conf, self.conf.cnn_dim, self.conf.cnn_kernal_size[i]))

        self.convs = nn.CellList(
            [nn.Conv2d(1, self.conf.cnn_out, (fsz, self.conf.cnn_dim), has_bias=True, pad_mode='valid') for fsz in
             self.conf.filter_sizes])  # cnn_out = 20; filter_sizes = [3, 4, 5];
        self.capsule = CatCapsuleModels.CapsuleNetwork(self.conf)

    #def construct(self, a1, a2, input_ids1, input_ids2, attention_mask1, attention_mask2, token_type_ids1,token_type_ids2, labels=None):
    def construct(self, a1, a2,labels=None):
        #  a1:128*50,这里从字典取出句子,token

        if a1.shape[1] > self.conf.max_sent_len:
            a1 = a1[:, :self.conf.max_sent_len]
        if a2.shape[1] > self.conf.max_sent_len:
            a2 = a2[:, :self.conf.max_sent_len]
        # print(a1)
        # print(a1.size())
        # print(a2)
        if self.conf.use_rnn:
            len1 = ms.Tensor([ops.max(ops.nonzero(a1[i, :]) + 1)[0].asnumpy() for i in range(a1.shape[0])], ms.int64)
            len2 = ms.Tensor([ops.max(ops.nonzero(a2[i, :]) + 1)[0].asnumpy() for i in range(a2.shape[0])], ms.int64)

        # 对句子进行最初的词嵌入 128*50*300
        arg1repr = self.embed(a1)
        arg2repr = self.embed(a2)
        # print(1111111111)
        # print(arg1repr[0,49,:])
        if self.conf.need_char or self.conf.need_sub: # False
            char1 = self.charenc(a1)
            char2 = self.charenc(a2)
            arg1repr = ops.cat((arg1repr, char1), axis=2)
            arg2repr = ops.cat((arg2repr, char2), axis=2)
        # 对应于论文中的LSTM，
        # if self.conf.need_bert:
        #     # arg1repr:128*50*400  bert提取100维度
        #     arg1repr = ops.cat((arg1repr, self.bert(input_ids1, attention_mask1, token_type_ids1)), axis=2)
        #     arg2repr = ops.cat((arg2repr, self.bert(input_ids2, attention_mask2, token_type_ids2)), axis=2)
        arg1repr = self.dropout(arg1repr)
        arg2repr = self.dropout(arg2repr)
        outputs = []
        reps = []
        for i in range(self.conf.cnn_layer_num):
            if self.conf.use_rnn:
                arg1repr = self.blocks[2 * i](arg1repr, len1)
                arg2repr = self.blocks[2 * i + 1](arg2repr, len2)
            else:
                #CNN卷积处理
                arg1repr = self.blocks[2 * i](arg1repr)
                arg2repr = self.blocks[2 * i + 1](arg2repr)

            #128*20*141
            output1 = self._cnn(arg1repr)
            output2 = self._cnn(arg2repr)
            #128*8000
            outputc1, outputc2, attnw = self.attn(arg1repr, arg2repr)
            reps.append(outputc1)
            reps.append(outputc2)
            output = ops.cat((output1, output2), 1)
            outputs.append(output)
        
        outputs = ops.cat(outputs, 2)
        self.capsule(outputs)

        reps = ops.cat(reps, 1)

        return self.capsule, reps

    def _cnn(self, rep):
        rep = rep.view(rep.shape[0], 1, rep.shape[1], rep.shape[2])
        outputs = [ops.relu(conv(rep)) for conv in self.convs]  # [128,20,(48,47,46),1]
        outputs = [x_item.view(x_item.shape[0], x_item.shape[1], -1) for x_item in outputs]  # [128,20,(48,47,46)]
        outputs = ops.cat(outputs, 2)

        return outputs


class Classifier(nn.Cell):
    def __init__(self, nclass, conf, model_name=''):
        super(Classifier, self).__init__()
        self.conf = conf
        self.dropout = nn.Dropout(p=self.conf.clf_dropout)
        self.fc = nn.CellList()
        if self.conf.clf_fc_num > 0:
            self.fc.append(nn.Dense(self.conf.pair_rep_dim, self.conf.clf_fc_dim, weight_init=Uniform(scale=0.01),
                                    bias_init="Zero"))
            for i in range(self.conf.clf_fc_num - 1):
                self.fc.append(nn.Dense(self.conf.clf_fc_dim, self.conf.clf_fc_dim, weight_init=Uniform(scale=0.01),
                                        bias_init="Zero"))
            lastfcdim = self.conf.clf_fc_dim
        else:
            lastfcdim = self.conf.pair_rep_dim

        if model_name in ['capsule']:
            if self.conf.need_bert:
                lastfcdim = 8000
            else:
                lastfcdim = 8000
        elif model_name in ['rnn_cnn']:
            lastfcdim = 600
        elif model_name in ['cat_capsule']:
            if self.conf.need_bert:
                lastfcdim = 120000

            else:
                lastfcdim = 80000
        elif model_name in ['tensor']:
            lastfcdim = 1280

        # self.lastfc = nn.Dense(lastfcdim, nclass, weight_init=Uniform(scale=0.01), bias_init="Zero")
        self.lastfc = nn.Dense(lastfcdim, nclass)
        self.nonlinear = nn.Tanh()

    def construct(self, input):
        output = input
        for i in range(self.conf.clf_fc_num):
            output = self.nonlinear(self.dropout(self.fc[i](output)))
        output = self.lastfc(self.dropout(output))
        return output


class BertLayer(nn.Cell):
    def __init__(self):
        super(BertLayer, self).__init__()
        #lookup:torch.Size([32262, 50])   lenth:torch.Size([32262, 1])


        #self.modelConfig = BertConfig.from_pretrained('./model1/dl_model/config.json')
        # self.model = BertModel.from_pretrained('./model1/dl_model/pytorch_model.bin', config=self.modelConfig)
        # self.fc = nn.Linear(768,100)
        self.model = BertModel.from_pretrained("bert-base-uncased")
        #self.model = BertModel.from_pretrained('/mnt/data/shw/CapsuleIDRR-bert-ms/model1/dl_model/pytorch_model.ckpt')
        self.fc = nn.Dense(768, 100, weight_init=Uniform(scale=0.01), bias_init="Zero")

    def construct(self, input_ids, attention_mask, token_type_ids):


        # text_dict = self.tokenizer.encode_plus(input, add_special_tokens=True,
        #                                   return_attention_mask=True, truncation=True)
        #
        # input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
        #
        # token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
        #
        # attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)

        for param in self.model.get_parameters():
            param.requires_grad = False

        print('-----')

        res = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        res = self.fc(res[0])
        print('-------')
        # return res[0].detach().squeeze(0)
        return res
