import logging
import time
from datetime import datetime
import pickle
#from sklearn.metrics import f1_score
from mindspore.train import Accuracy, F1
# from tensorboardX import SummaryWriter
from data import Data
from model import *
# from new_model import *
from optim_schedule import ScheduledOptim

import mindspore as ms
from mindspore import nn
from mindspore import ops


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ModelBuilder(object):
    def __init__(self, use_cuda, conf, model_name):
        self.cuda = use_cuda
        self.conf = conf
        self.model_name = model_name
        self._init_log()
        self._pre_data()
        self._build_model()

    def _pre_data(self):
        print('pre data...')
        self.data = Data(self.cuda, self.conf)

    def _init_log(self):
        if self.conf.four_or_eleven == 2:
            filename = 'logs/train_' + datetime.now().strftime(
                '%B%d-%H_%M_%S') + '_' + self.model_name + self.conf.type + '_' + self.conf.i2senseclass[
                           self.conf.binclass]
        else:
            filename = 'logs/train_' + datetime.now().strftime(
                '%B%d-%H_%M_%S') + '_' + self.model_name + '_' + self.conf.type

        if self.conf.need_bert:
            filename += '_BERT'

        logging.basicConfig(
            filename=filename + '.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s: %(message)s',
            level=logging.DEBUG)

    def _build_model(self):
        print('loading embedding...')
        if self.conf.corpus_splitting == 1:
            pre = './data/processed/lin/'
        elif self.conf.corpus_splitting == 2:
            pre = './data/processed/ji/'
        elif self.conf.corpus_splitting == 3:
            pre = './data/processed/l/'
        with open(pre + 'we.pkl', 'rb') as file:
            we = pickle.load(file)
        char_table = None
        sub_table = None
        if self.conf.need_char or self.conf.need_elmo:
            with open(pre + 'char_table.pkl', 'rb') as file:
                char_table = pickle.load(file)
        with open(pre + 'char_table.pkl', 'rb') as file:
            char_table = pickle.load(file)
        if self.conf.need_sub:
            with open(pre + 'sub_table.pkl', 'rb') as file:
                sub_table = pickle.load(file)
        print('building model...')
        if self.model_name == 'cat_capsule':
            self.encoder = CatCapsuleEncoder(self.conf, we, char_table, sub_table, self.cuda)

        self.classifier = Classifier(self.conf.clf_class_num, self.conf, self.model_name)
        if self.conf.is_mttrain:
            self.conn_classifier = Classifier(self.conf.conn_num, self.conf, self.model_name)

        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

        # para_filter = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        # 尝试使用AdamW
        # self.e_optimizer = torch.optim.Adagrad(para_filter(self.encoder), self.conf.lr,
        #                                        weight_decay=self.conf.l2_penalty)
        self.e_optimizer = nn.Adagrad(self.encoder.trainable_params(), learning_rate=self.conf.lr,
                                      weight_decay=self.conf.l2_penalty)
        # self.e_optimizer_schedule = ScheduledOptim(self.e_optimizer, self.conf.transformer_dim, self.conf.warmup_steps,
        #                                            self.conf)
        # self.c_optimizer = torch.optim.Adagrad(para_filter(self.classifier), self.conf.lr,
        #                                        weight_decay=self.conf.l2_penalty)
        self.c_optimizer = nn.Adagrad(self.classifier.trainable_params(), learning_rate=self.conf.lr,
                                      weight_decay=self.conf.l2_penalty)
        if self.conf.is_mttrain:
            # self.con_optimizer = torch.optim.Adagrad(para_filter(self.conn_classifier), self.conf.lr,
            #                                          weight_decay=self.conf.l2_penalty)
            self.con_optimizer = nn.Adagrad(self.conn_classifier.trainable_params(), learning_rate=self.conf.lr,
                                            weight_decay=self.conf.l2_penalty)
        self.test_f1 = 0

    def _print_train(self, epoch, time, loss, acc, learning_rate):
        print('-' * 80)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | loss: {:10.5f} | acc: {:5.2f}% | lr: {:10.10f} |'.format(
                epoch, time, loss, acc * 100, learning_rate
            )
        )
        print('-' * 80)
        logging.debug('-' * 80)
        logging.debug(
            '| end of epoch {:3d} | time: {:5.2f}s | loss: {:10.5f} | acc: {:5.2f}% | lr: {:10.10f} |'.format(
                epoch, time, loss, acc * 100, learning_rate
            )
        )
        logging.debug('-' * 80)

    def _print_eval(self, task, loss, acc, f1):
        print(
            '| ' + task + ' loss {:10.5f} | acc {:5.2f}% | f1 {:5.2f}%'.format(loss, acc * 100, f1 * 100)
        )
        print('-' * 80)
        logging.debug(
            '| ' + task + ' loss {:10.5f} | acc {:5.2f}% | f1 {:5.2f}%'.format(loss, acc * 100, f1 * 100)
        )
        logging.debug('-' * 80)

    def _save_model(self, model, filename):
        ms.save_checkpoint(model, './weights/' + filename)

    def _load_model(self, model, filename):
        param_dict = ms.load_checkpoint('./weights/' + filename)
        ms.load_param_into_net(model, param_dict)

    def _train_one(self):
        self.encoder.set_train(True)
        self.classifier.set_train(True)
        if self.conf.is_mttrain:
            self.conn_classifier.set_train(True)
        total_loss = 0
        correct_n = 0

        # a1 [batch_size, id] [128x50]   sense [128] 代表了4个关系    arg1_sen list 128原句子
        # def forward_fn(a1, a2, sense, conn, input_ids1, input_ids2, attention_mask1, attention_mask2, token_type_ids1,
        #                token_type_ids2):
        def forward_fn(a1, a2, sense, conn):
            if self.model_name in ['cat_capsule']:
                targets = ms.Tensor(np.zeros((sense.shape[0], self.conf.four_or_eleven)), ms.int64)
                # capsule, repr = self.encoder(a1, a2, input_ids1, input_ids2, attention_mask1, attention_mask2,
                #                              token_type_ids1, token_type_ids2)  # capsule:5(rnn+3cnn)  repr:5rnn
                capsule, repr = self.encoder(a1, a2)
                # for i in targets.shape[0]:
                #     targets[i][sense[i]] = 1  # 独热target
                for i, target in enumerate(targets):
                    # 128*4
                    targets[i][sense[i]] = 1
                loss = capsule.loss(targets.float())  # 句子对的胶囊损失

                clone_logits = capsule.logits  # [128,numclass]
                '''
                output_sense = ops.argmax(clone_logits, 1)
                assert output_sense.shape == sense.shape
                tmp = ms.Tensor((output_sense == sense), ms.int64)
                correct_n = ops.sum(tmp)
                '''
            return loss, clone_logits

        grad_fn1 = ms.value_and_grad(fn=forward_fn, grad_position=None,
                                     weights=self.e_optimizer.parameters, has_aux=True)# change True
        grad_fn2 = ms.value_and_grad(fn=forward_fn, grad_position=None,
                                     weights=self.c_optimizer.parameters, has_aux=True)
        grad_fn3 = ms.value_and_grad(forward_fn, None, self.con_optimizer.parameters,
                                     has_aux=True) if self.conf.is_mttrain else None

        grad_fns = [grad_fn1, grad_fn2]
        optimizers = [self.e_optimizer, self.c_optimizer]

        if self.conf.is_mttrain:
            grad_fns.append(grad_fn3)
            optimizers.append(self.con_optimizer)

        # def train_step(a1, a2, sense, conn, grad_fns, optimizers, input_ids1, input_ids2, attention_mask1,
        #                attention_mask2, token_type_ids1, token_type_ids2):
        def train_step(a1, a2, sense, conn):

            # loss, correct_n = forward_fn(a1, a2, sense, conn, input_ids1, input_ids2, attention_mask1, attention_mask2,
            #                              token_type_ids1, token_type_ids2)
            '''(loss, logits), grads = forward_fn(a1, a2, sense, conn)'''
            for grad_fn, optimizer in zip(grad_fns, optimizers):
                # _, grads = grad_fn(a1, a2, sense, conn, input_ids1, input_ids2, attention_mask1, attention_mask2,
                #                    token_type_ids1, token_type_ids2)
                (loss, logits), grads = grad_fn(a1, a2, sense, conn)
                #print("type of grads: ", type(grads))
                #print(len(grads))

                # grads[0] = ms.Tensor(grads[0],dtype=ms.float32)
                grads = ops.clip_by_norm(grads, self.conf.grad_clip)
                optimizer(grads)
            return loss, logits


        train_size = self.data.train_size
        print("train_size: ", train_size)

        # for i, (a1, a2, sense, conn, arg1_sen, arg2_sen, input_ids1, input_ids2, attention_mask1, attention_mask2,
        #         token_type_ids1, token_type_ids2) in enumerate(self.data.train_loader):
        for i, (a1, a2, sense, conn, arg1_sen, arg2_sen) in enumerate(self.data.train_loader):
            # print(i)
            # print(a1.shape)
            # print(a2.shape)
            # print(sense)
            # print(conn)
            # print(arg1_sen.shape)
            # print(arg2_sen.shape)

            if self.conf.four_or_eleven == 2:  # 二元分类
                mask1 = (sense == self.conf.binclass)
                mask2 = (sense != self.conf.binclass)
                sense[mask1] = 1
                sense[mask2] = 0
            # loss, correct_n = train_step(a1, a2, sense, conn, grad_fns, optimizers, input_ids1, input_ids2,
            #                              attention_mask1, attention_mask2, token_type_ids1, token_type_ids2)
            loss, logits = train_step(a1, a2, sense, conn)

            output_sense = ops.argmax(logits, 1)
            assert output_sense.shape == sense.shape
            #tmp = ms.Tensor((output_sense == sense), ms.int64)
            tmp = ms.Tensor((output_sense == sense))
            #print(correct_n)
            #k = ops.sum(tmp)

            correct_n += ops.sum(tmp).asnumpy()
            #print(correct_n)
            total_loss += loss.asnumpy() * sense.shape[0]

            print("iteration: ", i)

        return total_loss / train_size, float(correct_n) / train_size

    def _train(self, pre):
        for epoch in range(self.conf.epochs):
            start_time = time.time()
            loss, acc = self._train_one()
            self._print_train(epoch, time.time() - start_time, loss, acc, self.conf.lr)

            dev_loss, dev_acc, dev_f1 = self._eval('dev')
            self._print_eval('dev', dev_loss, dev_acc, dev_f1)

            test_loss, test_acc, test_f1 = self._eval('test')
            self._print_eval('test', test_loss, test_acc, test_f1)

            if test_f1 > self.test_f1:
                save_name = self.model_name + '_' + self.conf.type + '_' + str(self.conf.four_or_eleven)
                if self.conf.need_bert:
                    save_name += '_BERT'
                if self.conf.four_or_eleven == 2:
                    save_name += '_' + self.conf.i2senseclass[self.conf.binclass]

                self._save_model(self.encoder, save_name + '_eparams.ckpt')
                self._save_model(self.classifier, save_name + '_cparams.ckpt')
                self.test_f1 = test_f1

    def train(self, pre):
        print('start training')
        print(self.conf.__dict__)
        print(self.encoder)
        print(self.classifier)
        logging.debug('start training')
        logging.debug(self.conf.__dict__)
        logging.debug(self.encoder)
        logging.debug(self.classifier)
        self._train(pre)
        print('training done')
        logging.debug('training done')

    def _eval(self, task):
        global f1
        self.encoder.set_train(False)
        self.classifier.set_train(False)
        total_loss = 0
        correct_n = 0
        if task == 'dev':
            data = self.data.dev_loader
            n = self.data.dev_size
        elif task == 'test':
            data = self.data.test_loader
            n = self.data.test_size
        else:
            raise Exception('wrong eval task')

        output_list = []
        gold_list = []
        logic_list=[]
        # for i, (a1, a2, sense1, sense2, arg1_sen, arg2_sen, input_ids1, input_ids2, attention_mask1, attention_mask2,
        #         token_type_ids1, token_type_ids2) in enumerate(data):
        for i, (a1, a2, sense1, sense2, arg1_sen, arg2_sen) in enumerate(data):
            try:
                if self.conf.four_or_eleven == 2:
                    mask1 = (sense1 == self.conf.binclass)
                    mask2 = (sense1 != self.conf.binclass)
                    sense1[mask1] = 1
                    sense1[mask2] = 0
                    mask0 = (sense2 == -1)
                    mask1 = (sense2 == self.conf.binclass)
                    mask2 = (sense2 != self.conf.binclass)
                    sense2[mask1] = 1
                    sense2[mask2] = 0
                    sense2[mask0] = -1

                if self.model_name in ['capsule', 'cat_capsule']:
                    targets = ms.Tensor(np.zeros((sense1.shape[0], self.conf.four_or_eleven)), ms.int64)
                    # for i in targets.shape[0]:
                    #     targets[i][sense1[i]] = 1
                    for i, target in enumerate(targets):
                        # 128*4
                        targets[i][sense1[i]] = 1
                    # -------
                    # -------
                    # capsule, _ = self.encoder(a1, a2, input_ids1, input_ids2, attention_mask1, attention_mask2,
                    #                           token_type_ids1, token_type_ids2, ms.Tensor(np.array([0, 1, 2, 3])))
                    capsule, _ = self.encoder(a1, a2, ms.Tensor(np.array([0, 1, 2, 3])))
                    loss = capsule.loss(targets.float())
                    clone_logits = capsule.logits
                    output_sense = ops.argmax(clone_logits, 1)
                else:
                    output = self.classifier(self.encoder(a1, a2))
                    _, output_sense = ops.max(output, 1)

                assert output_sense.shape == sense1.shape
                gold_sense = sense1
                mask = (output_sense == sense2)
                gold_sense[mask] = sense2[mask]
                #tmp = (output_sense == gold_sense)
                tmp = ms.Tensor((output_sense == gold_sense))
                correct_n += ops.sum(tmp).asnumpy()
                logic_list.append(clone_logits)
                output_list.append(output_sense)
                gold_list.append(gold_sense)

                if self.model_name not in ['capsule', 'cat_capsule']:
                    loss = self.criterion(output, gold_sense)

                total_loss += loss.asnumpy() * gold_sense.shape[0]
                logic_s = ops.cat(logic_list)
                output_s = ops.cat(output_list)
                #output_s = ops.cat(clone_logits)
                #output_s = output_s.unsqueeze(-1)
                k = output_s.asnumpy()
                gold_s = ops.cat(gold_list)


                if self.conf.four_or_eleven == 2:
                    f1_metric = F1()
                    f1_metric.update(gold_s.asnumpy(),output_s.asnumpy())
                    f1 = f1_metric.eval()
                    #f1 = F1(gold_s.asnumpy(), output_s.asnumpy(), average='binary')
                else:
                    #f1 = F1(gold_s.asnumpy(), output_s.asnumpy(), average='macro')
                    f1_metric = F1()
                    #f1_metric.update(output_s.asnumpy(), gold_s.asnumpy())
                    f1_metric.update(logic_s.asnumpy(), gold_s.asnumpy())
                    f1 = f1_metric.eval(average=True)

            except Exception as e:
                print(e)
                logging.debug(e)
                continue

        self.encoder.set_train(True)
        self.classifier.set_train(True)
        return total_loss / n, float(correct_n) / n, f1

    def eval(self, pre):
        print('evaluating...')
        logging.debug('evaluating...')
        save_name = self.model_name + '_' + self.conf.type + '_' + str(self.conf.four_or_eleven)
        if self.conf.need_bert:
            save_name += '_BERT'
        if self.conf.four_or_eleven == 2:
            save_name = save_name + '_' + self.conf.i2senseclass[self.conf.binclass]
        pre = save_name
        self._load_model(self.encoder, pre + '_eparams.ckpt')
        self._load_model(self.classifier, pre + '_cparams.ckpt')
        test_loss, test_acc, f1 = self._eval('test')
        self._print_eval('test', test_loss, test_acc, f1)
