"""
Author: Yunpeng Chen
"""
import os
import time
import socket
import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import shutil
from tensorboardX import SummaryWriter

from . import metric
from . import callback

"""
Static Model
"""
class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 **kwargs):
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        # init params
        self.net = net
        self.start_epoch = 0
        self.model_prefix = model_prefix
        self.criterion = criterion

    def load_state(self, state_dict, strict=False):
        if strict:
            self.net.load_state_dict(state_dict=state_dict)
        else:
            # customized partialy load function
            net_state_keys = list(self.net.state_dict().keys())
            for name, param in state_dict.items():
                if name in self.net.state_dict().keys():
                    dst_param_shape = self.net.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.net.state_dict()[name].copy_(param.view(dst_param_shape))
                        net_state_keys.remove(name)
            # indicating missed keys
            if net_state_keys:
                logging.warning(">> Failed to load: {}".format(net_state_keys))
                return False
        return True

    def get_checkpoint_path(self, epoch):
        assert self.model_prefix, "model_prefix undefined!"
        if torch.distributed._initialized:
            hostname = socket.gethostname()
            checkpoint_path = "{}_at-{}_ep-{:04d}.pth".format(self.model_prefix, hostname, epoch)
        else:
            checkpoint_path = "{}_ep-{:04d}.pth".format(self.model_prefix, epoch)
        return checkpoint_path

    def load_checkpoint(self, epoch, optimizer=None):

        load_path = self.get_checkpoint_path(epoch)
        assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

        checkpoint = torch.load(load_path)

        all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)

        if optimizer:
            if 'optimizer' in checkpoint.keys() and all_params_matched:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
            else:
                logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
        else:
            logging.info("Only model state resumed from: `{}'".format(load_path))

        if 'epoch' in checkpoint.keys():
            if checkpoint['epoch'] != epoch:
                logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

    def save_checkpoint(self, epoch, optimizer_state=None):

        save_path = self.get_checkpoint_path(epoch)
        save_folder = os.path.dirname(save_path)

        if not os.path.exists(save_folder):
            logging.debug("mkdir {}".format(save_folder))
            os.makedirs(save_folder)

        if not optimizer_state:
            torch.save({'epoch': epoch,
                        'state_dict': self.net.state_dict()},
                        save_path)
            logging.info("Checkpoint (only model) saved to: {}".format(save_path))
        else:
            torch.save({'epoch': epoch,
                        'state_dict': self.net.state_dict(),
                        'optimizer': optimizer_state},
                        save_path)
            logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))

    def unlabeled_weight(self, epoch, t1, t2, af):
        if epoch == 0 and self.callback_kwargs['batch'] == 0:
            logging.info("t1: {}, t2: {}, af: {}".format(t1, t2, af))
        alpha = 0.0
        if epoch > t1:
            alpha = (epoch - t1) / (t2 - t1) * af
            if epoch > t2:
                alpha = af
        return alpha


    def forward(self, true_data, true_target, epoch=0, val_acc=[]):
        """ typical forward function with:
            single output and single loss
        """

        true_data = true_data.float().cuda()
        true_target = true_target.cuda()

        softmax = torch.nn.Softmax()

        if self.net.training:
            cond = (true_target >= 0)
            nnz = torch.nonzero(cond)
            labeled_bs = len(nnz)

            input_var = Variable(true_data, requires_grad=False)
            target_var = Variable(true_target, requires_grad=False)

            true_output = self.net(input_var)

            if hasattr(self, 'criterion') and self.criterion is not None \
                    and true_target is not None:

                labeled_loss = torch.sum(self.criterion(true_output, target_var)) / labeled_bs if labeled_bs > 0 else 0

                # with torch.no_grad():
                #     pseudo_labeled = true_output.max(1)[1]
                #     pseudo_labeled[nnz.view(-1)] = -1
                #
                #     test = softmax(true_output)
                #     test1 = -np.sort(-test.cpu().numpy(), 1)
                #
                #     # 未标注数据最大预测概率小于0.2直接pass
                #     for i in range(test1.shape[0]):
                #         # if self.start_epoch != 0:
                #         #     print(test1[i][:3])
                #         # if test1[i][0] < 0.6:
                #         if test1[i][0] - test1[i][1] < 0.2:
                #         # if test1[i][1] + test1[i][2] > 0.1:
                #             pseudo_labeled[i] = -1
                #
                #     unlabeled_bs = len(torch.nonzero(pseudo_labeled >= 0))
                #
                # # unlabeled_loss = torch.sum(target_var.eq(-1).float() * self.criterion(true_output, pseudo_labeled)) / (true_data.size(0)-labeled_bs +1e-10)
                # unlabeled_loss = torch.sum(self.criterion(true_output, pseudo_labeled)) / (unlabeled_bs + 1e-10)
                #
                # w = 0
                # if epoch > 2:
                #     if sum(val_acc[epoch - 3: epoch]) / len(val_acc[epoch - 3: epoch]) > 0.20000 and self.start_epoch == 0:
                #         self.start_epoch = epoch
                #     if self.start_epoch != 0:
                #         w = self.unlabeled_weight(epoch, self.start_epoch, self.start_epoch + 20, 1)
                # loss = labeled_loss + w * unlabeled_loss

                w = 0
                loss = labeled_loss

            else:
                loss = None

            return true_output, target_var, loss, w

        else:
            input_var = Variable(true_data, volatile=True)
            target_var = Variable(true_target, volatile=True)

            true_output = self.net(input_var)

            if hasattr(self, 'criterion') and self.criterion is not None \
                    and true_target is not None:
                loss = self.criterion(true_output, target_var)
            else:
                loss = None

            return true_output, loss


"""
Dynamic model that is able to update itself
"""
class model(static_model):

    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback=None,
                 step_callback_freq=50,
                 epoch_callback=None,
                 save_checkpoint_freq=1,
                 opt_batch_size=None,
                 **kwargs):

        # load parameters
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        super(model, self).__init__(net, criterion=criterion,
                                         model_prefix=model_prefix)

        # load optional arguments
        # - callbacks
        self.callback_kwargs = {'epoch': None,
                                'batch': None,
                                'sample_elapse': None,
                                'update_elapse': None,
                                'epoch_elapse': None,
                                'namevals': None,
                                'optimizer_dict': None,}

        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(),
                                                  callback.MetricPrinter())
        if not epoch_callback:
            epoch_callback = (lambda **kwargs: None)

        self.step_callback = step_callback
        self.step_callback_freq = step_callback_freq
        self.epoch_callback = epoch_callback
        self.save_checkpoint_freq = save_checkpoint_freq
        self.batch_size=opt_batch_size


    """
    In order to customize the callback function,
    you will have to overwrite the functions below
    """
    def step_end_callback(self):
        # logging.debug("Step {} finished!".format(self.i_step))
        self.step_callback(**(self.callback_kwargs))

    def epoch_end_callback(self):
        self.epoch_callback(**(self.callback_kwargs))
        if self.callback_kwargs['epoch_elapse'] is not None:
            logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                    self.callback_kwargs['epoch'],
                    self.callback_kwargs['epoch_elapse'],
                    self.callback_kwargs['epoch_elapse']/3600.))
        if self.callback_kwargs['epoch'] == 0 \
           or ((self.callback_kwargs['epoch']+1) % self.save_checkpoint_freq) == 0:
            self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1,
                                 optimizer_state=self.callback_kwargs['optimizer_dict'])

    """
    Learning rate
    """
    def adjust_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = lr * lr_mult

    """
    Optimization
    """
    def fit(self,
            train_iter,
            optimizer,
            lr_scheduler,
            eval_iter=None,
            metrics=metric.Accuracy(topk=1),
            epoch_start=0,
            epoch_end=10000,
            **kwargs):

        """
        checking
        """
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.cuda.is_available(), "only support GPU version"

        """
        start the main loop
        """
        pause_sec = 0.
        w = 0
        val_acc = []

        log_dir = './logs/hmdb51/move_style1'
        if os.path.exists(log_dir):
            shutil.rmtree(path=log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        for i_epoch in range(epoch_start, epoch_end):
            self.callback_kwargs['epoch'] = i_epoch
            epoch_start_time = time.time()

            # w = weight_schedule(i_epoch, epoch_end, 30., -5., 510, n_samples)
            # print('unsupervised loss weight : {}'.format(w))
            # w = Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)

            ###########
            # 1] TRAINING
            ###########
            metrics.reset()
            self.net.train()
            sum_sample_inst = 0
            sum_sample_elapse = 0.
            sum_update_elapse = 0
            batch_stop = 0
            batch_start_time = time.time()
            loss_list = []
            logging.info("Start epoch {:d}:".format(i_epoch))
            logging.info("The current value of w is {}".format(w))
            # for i_batch, (true_data, true_target) in enumerate(train_iter):
            for i_batch, (true_data, true_target) in enumerate(train_iter):
                self.callback_kwargs['batch'] = i_batch

                update_start_time = time.time()

                # output, loss, w = self.forward(true_data, true_target, epoch=i_epoch, val_acc=val_acc)
                output, true_target, loss, w = self.forward(true_data, true_target, epoch=i_epoch, val_acc=val_acc)

                # [backward]
                if loss is not None and loss != 0:
                    loss_list.append(loss.data.cpu().detach())
                    optimizer.zero_grad()
                    loss.backward()
                    self.adjust_learning_rate(optimizer=optimizer,
                                              lr=lr_scheduler.update())
                    optimizer.step()

                    # [evaluation] update train metric
                    metrics.update([output.data.cpu()],
                                   true_target.cpu(),
                                   [loss.data.cpu()])

                # timing each batch
                sum_sample_elapse += time.time() - batch_start_time
                sum_update_elapse += time.time() - update_start_time
                batch_start_time = time.time()
                sum_sample_inst += true_data.shape[0]

                if (i_batch % self.step_callback_freq) == 0:
                    # retrive eval results and reset metic
                    self.callback_kwargs['namevals'] = metrics.get_name_value()
                    metrics.reset()
                    # speed monitor
                    self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                    self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
                    sum_update_elapse = 0
                    sum_sample_elapse = 0
                    sum_sample_inst = 0
                    # callbacks
                    self.step_end_callback()

            ###########
            # 2] END OF EPOCH
            ###########

            writer.add_scalar('train_loss', sum(loss_list) / len(loss_list), i_epoch)

            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
            self.epoch_end_callback()

            ###########
            # 3] Evaluation
            ###########
            if (eval_iter is not None) \
                and ((i_epoch+1) % max(1, int(self.save_checkpoint_freq/2))) == 0:
                logging.info("Start evaluating epoch {:d}:".format(i_epoch))

                metrics.reset()
                self.net.eval()
                sum_sample_elapse = 0.
                sum_sample_inst = 0
                sum_forward_elapse = 0.
                batch_start_time = time.time()
                for i_batch, (data, target) in enumerate(eval_iter):
                    self.callback_kwargs['batch'] = i_batch

                    forward_start_time = time.time()

                    output, loss = self.forward(data, target)

                    metrics.update([output.data.cpu()],
                                    target.cpu(),
                                   [loss.data.cpu()])

                    sum_forward_elapse += time.time() - forward_start_time
                    sum_sample_elapse += time.time() - batch_start_time
                    batch_start_time = time.time()
                    sum_sample_inst += data.shape[0]

                # evaluation callbacks
                self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                self.callback_kwargs['update_elapse'] = sum_forward_elapse / sum_sample_inst
                self.callback_kwargs['namevals'] = metrics.get_name_value()
                self.step_end_callback()

                val_acc_batch = metrics.get_name_value()[1][0][1]
                writer.add_scalar('val_acc', val_acc_batch, i_epoch)
                val_acc.append(val_acc_batch)

        logging.info("Optimization done!")
