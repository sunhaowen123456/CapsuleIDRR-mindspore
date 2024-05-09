import mindspore as ms
import pickle
from mindnlp.transformers import BertTokenizer
from model import *

class Dataset(object):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        # self.d1, self.d2, self.d3, self.d4, self.d5, self.d6 = torch.load(data_path)
        with open(data_path, 'rb') as file:
            self.d1, self.d2, self.d3, self.d4, self.d5, self.d6 = pickle.load(file)
        self.d1, self.d2, self.d3, self.d4 = self.d1.asnumpy(), self.d2.asnumpy(), self.d3.asnumpy(), self.d4.asnumpy()

        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.tokenizer = BertTokenizer.from_pretrained('./model1/dl_model/vocab.txt')



    def __getitem__(self, index):
        index = int(index)
        # text_dict1 = self.tokenizer.encode_plus(' '.join(self.d5[0][index]),return_tensors='ms',add_special_tokens=True,
        #                                         return_attention_mask=True, padding='max_length',max_length=50, truncation=True)
        # text_dict2 = self.tokenizer.encode_plus(' '.join(self.d6[0][index]), return_tensors='ms',add_special_tokens=True,
        #                                         return_attention_mask=True, padding='max_length', max_length=50, truncation=True)

        # print(torch.tensor(text_dict1['input_ids']))
        # print('--------')
        # print(self.d1[index])
        #, torch.tensor(text_dict1['input_ids']), torch.tensor(text_dict2['input_ids'])
        # for key, value in text_dict1.items():
        #     #text_dict1[key] = torch.tensor(value).clone().detach()
        #     text_dict1[key] = ms.Tensor.from_numpy(np.array(value))
        #
        # for key,value in text_dict2.items():
        #     #text_dict2[key] = torch.tensor(value).clone().detach()
        #     text_dict2[key] = ms.Tensor.from_numpy(np.array(value))
        #k = text_dict1['input_ids1'].shape
        # for i in range(k):
        #     for j in range(p):
        #         text_dict1['input_ids'][i][j] = int(5000) if text_dict1['input_ids'][i][j] > 5000 else text_dict1['input_ids'][i][j]
        #         text_dict2['input_ids'][i][j] = int(5000) if text_dict1['input_ids'][i][j] > 5000 else \
        #         text_dict2['input_ids'][i][j]
        # for i in range(k):
        #
        #         text_dict1['input_ids'][i] = int(5000) if text_dict1['input_ids'][i]> 5000 else text_dict1['input_ids'][i]
        #         text_dict2['input_ids'][i] = int(5000) if text_dict1['input_ids'][i] > 5000 else \
        #         text_dict2['input_ids'][i]

        return (self.d1[index], self.d2[index], self.d3[index], self.d4[index], ' '.join(self.d5[0][index]), ' '.join(self.d6[0][index]))
        #return self.d1[index], self.d2[index], self.d3[index], self.d4[index], ' '.join(self.d5[0][index]), ' '.join(self.d6[0][index]), text_dict1['input_ids'], text_dict2['input_ids'], text_dict1['attention_mask'], text_dict2['attention_mask'],text_dict1['token_type_ids'], text_dict2['token_type_ids']


    def __len__(self):
        return len(self.d4)

class Data(object):
    def __init__(self, use_cuda, conf, batch_size=None):
        if batch_size is None:
            batch_size = conf.batch_size
        kwargs = {'batch_size': batch_size, 'shuffle': conf.shuffle, 'drop_last': False}
        if use_cuda:
            kwargs['pin_memory'] = True
        if conf.corpus_splitting == 1:
            pre = './data/processed/lin/'
        elif conf.corpus_splitting == 2:
            pre = './data/processed/ji/'
        elif conf.corpus_splitting == 3:
            pre = './data/processed/l/'
        train_data = Dataset(pre+'train.pkl')
        dev_data = Dataset(pre+'dev.pkl')
        test_data = Dataset(pre+'test.pkl')
        self.train_size = len(train_data)
        self.dev_size = len(dev_data)
        self.test_size = len(test_data)
        # self.train_loader = torchdata.DataLoader(train_data, **kwargs)
        # self.dev_loader = torchdata.DataLoader(dev_data, **kwargs)
        # self.test_loader = torchdata.DataLoader(test_data, **kwargs)
        # train_loader = ms.dataset.GeneratorDataset(train_data, column_names=["d1", "d2", "d3", "d4", "d5", "d6","input_ids1", "input_ids2", "attention_mask1", "attention_mask2",
        #         "token_type_ids1", "token_type_ids2"],shuffle=kwargs['shuffle'])
        # dev_loader = ms.dataset.GeneratorDataset(dev_data, column_names=["d1", "d2", "d3", "d4", "d5", "d6","input_ids1", "input_ids2", "attention_mask1", "attention_mask2",
        #         "token_type_ids1", "token_type_ids2"],shuffle=kwargs['shuffle'])
        # test_loader = ms.dataset.GeneratorDataset(test_data, column_names=["d1", "d2", "d3", "d4", "d5", "d6","input_ids1", "input_ids2", "attention_mask1", "attention_mask2",
        #         "token_type_ids1", "token_type_ids2"],shuffle=kwargs['shuffle'])
        train_loader = ms.dataset.GeneratorDataset(train_data,
                                                   column_names=["d1", "d2", "d3", "d4", "d5", "d6", ],
                                                   shuffle=kwargs['shuffle'])
        dev_loader = ms.dataset.GeneratorDataset(dev_data,
                                                 column_names=["d1", "d2", "d3", "d4", "d5", "d6", ],
                                                 shuffle=kwargs['shuffle'])
        test_loader = ms.dataset.GeneratorDataset(test_data,
                                                  column_names=["d1", "d2", "d3", "d4", "d5", "d6", ],
                                                  shuffle=kwargs['shuffle'])

        self.train_loader = train_loader.batch(batch_size=kwargs['batch_size'], drop_remainder=kwargs['drop_last'])
        self.dev_loader = dev_loader.batch(batch_size=kwargs['batch_size'], drop_remainder=kwargs['drop_last'])
        self.test_loader = test_loader.batch(batch_size=kwargs['batch_size'], drop_remainder=kwargs['drop_last'])

def abc():
    from config import Config
    # data = Data(False, Config())
    # print(data.train_size)
    # print(data.dev_size)
    # print(data.test_size)
    # for data in data.train_loader:
    #     print(data)
    #loaders = [data.train_loader, data.dev_loader, data.test_loader]
    data1 = Data(False, Config())
    train_size = data1.train_size
    print(train_size) # 12775
    for i, (a1, a2, sense, conn, arg1_sen, arg2_sen) in enumerate(data1.train_loader):
        print(i)
        print(type(a1))
        print(type(a2))
        print(type(sense))
        print(conn)
        print(type(arg1_sen))
        print(type(arg2_sen))
        break
        # if i == 5:
        #     break

    print(1111)
    # for i in range(k):
    #     for j in range(p):
    #         input_ids1[i][j]=int(5000) if input_ids1[i][j] > 5000 else input_ids1[i][j]
    #elmo = ElmoLayer(char_table=torch.load('./data/processed/l/'+ 'char_table.pkl'),conf=Config())
    #bert = new_model.BertLayer()
    # print(type(a1_bert))
    # print(a1_bert['input_ids'])
    #rep = elmo(a1)
    #print(len(rep))
    #print(rep.shape)
    # bert = BertModel.from_pretrained('bert-base-uncased')
    # rep = bert(input_ids1, attention_mask1, token_type_ids1)
    # print(rep[0])
    #
    # print(len(rep))
    # for loader in loaders:
    #     res = {}
    #     for d in loader:
    #         l = []
    #         for i in d:
    #             l.append(i.size())
    #         if tuple(l) not in res:
    #             res[tuple(l)] = 1
    #         else:
    #             res[tuple(l)] += 1
    #     for i in res:
    #         print(i, '*', res[i])
    #     print('-' * 100)


if __name__ == '__main__':
    abc()
