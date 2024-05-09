import argparse
import os

import mindspore as ms
from builder import ModelBuilder
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--func', type=str, default='train')
parser.add_argument('-m', '--model', type=str, default='cat_capsule')
parser.add_argument('-s', '--splitting', type=int, default=3)


def main(conf, is_train=True, model_name='cat_capsule', pre=None):
    # havecuda = torch.cuda.is_available()
    havecuda = False
    ms.set_seed(conf.seed)
    # ms.set_context(device_target="GPU",
    #                device_id=0,
    #                mode=ms.PYNATIVE_MODE,
    #                pynative_synchronize=True)
    ms.set_context(device_target="GPU",
                   device_id=0)

    model = ModelBuilder(havecuda, conf, model_name)
    if is_train:
        model.train(pre)
    else:
        model.eval(pre)


if __name__ == '__main__':
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    A = parser.parse_args()
    if A.splitting == 3:
        conf = Config(4, 3, A.model)
        if A.func == 'train':
            main(conf, True, A.model, 'l')
        elif A.func == 'eval':
            main(conf, False, A.model, 'l')
        else:
            raise Exception('wrong fun')
    print(1)
