import torch
from torch import nn
import logging
import time
import os
from torchvision import transforms

from metric import AverageMeter
from model import Network
from model import Compress_and_DeCompress
from dataset import dataLoader

logging.basicConfig(filename="./train_info.log", level=logging.INFO)
logging.getLogger()


def train(net, rd_lambda, train_iter, num_epochs, loss, device, \
          log_interval=100, update_interval=30):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_time, rates, distortions, rd_loss = [AverageMeter(log_interval) for _ in range(4)]

    logging.info("The followings are train results:")
    count = 0

    for epoch in range(num_epochs):
        net.train()
        start_time = time.time()
        logging.info(f'Epoch: {epoch}')
        # count = 1
        for i, X in enumerate(train_iter):
            # timer.start()
            count += 1
            optimizer.zero_grad()
            X = X.to(device)
            X_hat, rate = net(X)
            l, distortion = loss(rate, rd_lambda, X, X_hat)
            l.backward()

            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)

            clip_gradient(optimizer, 5)

            optimizer.step()
            # print("loss: ", l)

            if (count % update_interval) == 0:
                # 每隔几十次更新一次数据
                train_time.update(time.time() - start_time)
                start_time = time.time()
                rates.update(rate)
                distortions.update(distortion)
                rd_loss.update(l)

            if (count % log_interval) == 0:
                # 每隔几十次打印一次数据
                info = " | ".join([f'Epoch: {epoch}',
                                   f'update_interval: {train_time.avg} s',
                                   f'rate: {rates.avg}',
                                   f'distortion(mse): {distortions.avg}',
                                   f'rd_loss: {rd_loss.avg}'])
                logging.info(info)

            # if epoch == num_epochs - 1:
            #     # 转成图片
            #     toPIL = transforms.ToPILImage()
            #     # for i in range(X_hat.shape[0]):
            #     #     print(X_hat[i] * 255, X[i] * 255)
            #     #     img = toPIL(X_hat[i] * 255)
            #     #     img.save(os.path.join(os.getcwd(), 'image/train_results', str(epoch)+'-'+str(count)+'-'+str(i)+'.bmp'))                  
            #     for i in range(X_hat.shape[0]):
            #         print(X_hat[i], X[i])
            #         img = toPIL(X_hat[i].clamp(0., 1.))
            #         img.save(os.path.join(os.getcwd(), 'image', 'train_results',
            #                               str(epoch) + '-' + str(count) + '-' + str(i) + '.bmp'))



def test(net, rd_lambda, test_iter, loss, device):

    print('test on', device)
    net.to(device)
    net.eval()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_time, rates, distortions, rd_loss = [AverageMeter(1) for _ in range(4)]

    logging.info("The followings are test result:")
    
    start_time = time.time()
    for num, X in enumerate(test_iter):

        X = X.to(device)
        X_hat, rate = net(X)
        l, distortion = loss(rate, rd_lambda, X, X_hat)

        train_time.update(time.time() - start_time)
        start_time = time.time()
        rates.update(rate)
        distortions.update(distortion)
        rd_loss.update(l)


        info = " | ".join([
                            f'update_interval: {train_time.avg} s',
                            f'rate: {rates.avg}',
                            f'distortion(mse): {distortions.avg}',
                            f'rd_loss: {rd_loss.avg}'])
        logging.info(info)

        # 转成图片
        toPIL = transforms.ToPILImage()
        # for i in range(X_hat.shape[0]):
        #     print(X_hat[i] * 255, X[i] * 255)
        #     img = toPIL(X_hat[i] * 255)
        #     img.save(os.path.join(os.getcwd(), 'image/train_results', str(epoch)+'-'+str(count)+'-'+str(i)+'.bmp'))                  
        for i in range(X_hat.shape[0]):
            img = toPIL(X_hat[i].clamp(0., 1.))
            img.save(os.path.join(os.getcwd(), 'image', 'test_results',
                                    str(num) + '-' + str(i) + '.bmp'))


def loss(rate, rd_lambda, X, X_hat):
    distortion = torch.mean(torch.square(X - X_hat))
    # print(torch.square(X - X_hat)[0][0][120][120])
    # print(distortion)
    return rate + rd_lambda * distortion, distortion


if __name__ == "__main__":
    #torch.set_printoptions(edgeitems=20)
    train_iter = dataLoader("/root/autodl-tmp/train2017", batch_size=8)
    #train_iter = dataLoader(os.path.join(os.getcwd(), "image/train"), batch_size=10)
    test_iter = dataLoader(os.path.join(os.getcwd(), "image/test"), batch_size=1)
    #net = nn.Sequential(Compress_and_DeCompress(training=True, device='cuda:0'))
    net = Network(device='cuda:0')
    train(net=net, rd_lambda=1024, train_iter=train_iter, num_epochs=150, loss=loss, device='cuda:0', log_interval=10, update_interval=2)
    torch.save(net.state_dict(), os.path.join(os.getcwd(), "model_param", "model1024"))
    test(net=net, rd_lambda=1024, test_iter=test_iter, loss=loss, device='cuda:0')

