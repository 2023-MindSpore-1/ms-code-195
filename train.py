# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
train
code: https://github.com/google/automl/tree/master/efficientnetv2
paper: https://arxiv.org/abs/2104.00298
Acc: ImageNet1k-84.9% (pretrained on ImageNet22k)
"""
import os

from mindspore import Model, nn, context, set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.args import args
from src.tools.callback import ModelArtsCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer
import numpy as np
import tqdm

import mindspore.ops as ops
from mindspore.nn import Cell
import mindspore as ms
from collections.abc import Iterable

_sum_op = ops.MultitypeFuncGraph("grad_sum_op")
_clear_op = ops.MultitypeFuncGraph("clear_op")

@_sum_op.register("Tensor", "Tensor")
def _cumulative_grad(grad_sum, grad):
    """Apply grad sum to cumulative gradient."""
    add = ops.AssignAdd()
    return add(grad_sum, grad)

@_clear_op.register("Tensor", "Tensor")
def _clear_grad_sum(grad_sum, zero):
    """Apply zero to clear grad_sum."""
    success = True
    success = ops.depend(success, ops.assign(grad_sum, zero))
    return success

class TrainForwardBackward(Cell):
    def __init__(self, network, optimizer, grad_sum, sens=1.0):
        super(TrainForwardBackward, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad_sum = grad_sum
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.hyper_map = ops.HyperMap()

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        return ops.depend(loss, self.hyper_map(ops.partial(_sum_op), self.grad_sum, grads))

class TrainOptim(Cell):
    def __init__(self, optimizer, grad_sum):
        super(TrainOptim, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.grad_sum = grad_sum

    def construct(self):
        return self.optimizer(self.grad_sum)

class TrainClear(Cell):
    def __init__(self, grad_sum, zeros):
        super(TrainClear, self).__init__(auto_prefix=False)
        self.grad_sum = grad_sum
        self.zeros = zeros
        self.hyper_map = ops.HyperMap()

    def construct(self):
        success = self.hyper_map(ops.partial(_clear_op), self.grad_sum, self.zeros)
        return success

class GradientAccumulation:
    def __init__(self, network, loss_fn, optimizer):
        self._network = network
        self._loss_fn = loss_fn
        self._optimizer = optimizer

        params = self._optimizer.parameters
        self._grad_sum = params.clone(prefix="grad_sum", init='zeros')
        self._zeros = params.clone(prefix="zeros", init='zeros')
        self._train_forward_backward = self._build_train_forward_backward_network()
        self._train_optim = self._build_train_optim()
        self._train_clear = self._build_train_clear()

    @staticmethod
    def _transform_callbacks(callbacks):
        """Transform callback to a list."""
        if callbacks is None:
            return []

        if isinstance(callbacks, Iterable):
            return list(callbacks)

        return [callbacks]

    def _build_train_forward_backward_network(self):
        """Build forward and backward network"""
        network = self._network
        network = nn.WithLossCell(network, self._loss_fn)
        loss_scale = 1.0
        network = TrainForwardBackward(network, self._optimizer, self._grad_sum, loss_scale).set_train()
        return network

    def _build_train_optim(self):
        """Build optimizer network"""
        network = TrainOptim(self._optimizer, self._grad_sum).set_train()
        return network

    def _build_train_clear(self):
        """Build clear network"""
        network = TrainClear(self._grad_sum, self._zeros).set_train()
        return network

    def train_process(self, epoch, train_dataset, mini_steps=None):
        """
        Training process. The data would be passed to network directly.
        """
        dataset_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=False, epoch_num=epoch)

        for i in range(epoch):
            step = 0
            for k, next_element in enumerate(dataset_helper):
                loss = self._train_forward_backward(*next_element)
                print(loss)
                if (k + 1) % mini_steps == 0:
                    step += 1
                    print("epoch:", i + 1, "step:", step, "loss is ", loss)
                    self._train_optim()
                    self._train_clear()

            train_dataset.reset()

        ms.save_checkpoint(self._train_forward_backward, "gradient_accumulation.ckpt", )


def binary_accuracy(preds, y):
    """
    计算每个batch的准确率
    """

    # 对预测值进行四舍五入
    rounded_preds = np.around(preds)
    correct = (rounded_preds == y).astype(np.float32)
    acc = correct.sum() / len(correct)
    return acc



def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    loss_total = 0
    step_total = 0
    # with tqdm.tqdm(total=total) as t:
    #     t.set_description('Epoch %i' % epoch)
    for i in train_dataset.create_tuple_iterator():
        loss = model(*i)
        loss_total += loss.asnumpy()
        step_total += 1
        print(loss)
            # t.set_postfix(loss=loss_total/step_total)
            # t.update(1)

def main():
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    # rank = set_device(args)

    # get model and cast amp_level
    net = get_model(args)
    # print(net)
    cast_amp(net)


    criterion = get_criterion(args)
    # if args.pretrained:
    #     pretrained(args, net)
    net_with_loss = NetWithLoss(net, criterion)
    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()

    optimizer = get_optimizer(args, net, batch_num)
    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    # net_with_loss = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer)

    total_step = 4
    drop_rates = np.linspace(0.1, 0.3, total_step)
    magnitudes = np.linspace(5, 15, total_step)

    for step in range(total_step):
        net_with_loss.network.model.drop_rate = drop_rates[step]
        ratio = float(step+1)/total_step
        start_epoch = int(step/total_step * args.epochs)
        end_epoch = int(ratio * args.epochs)
        args.input_image_size = int(128 + (args.image_size-128) * ratio)

        data = get_dataset(args)
        model = GradientAccumulation(net, criterion, optimizer)
        for epoch in range(start_epoch, end_epoch):
            model.train_process(10, data.train_dataset, mini_steps=4)
            # train_one_epoch(net_with_loss, data.train_dataset, epoch)
            # valid_loss = evaluate(net, imdb_valid, loss, epoch)

            # if valid_loss < best_valid_loss:
            #     best_valid_loss = valid_loss
            #     ms.save_checkpoint(net, ckpt_file_name)

        del data

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)
    ckpt_save_dir = "./ckpt_"
    if args.run_modelarts:
        ckpt_save_dir = "/cache/ckpt_" + str(rank)
    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=args.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    ckpoint_cb = ModelCheckpoint(prefix=args.arch, directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    callbacks = [ckpoint_cb, loss_cb, time_cb]
    if args.run_modelarts:
        modelarts_cb = ModelArtsCallBack(src_url=ckpt_save_dir, save_freq=args.save_every,
                                         train_url=os.path.join(args.train_url, "ckpt_"))
        callbacks.append(modelarts_cb)

    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset, callbacks=callbacks, dataset_sink_mode=True)
    print("train success")

    if args.run_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=ckpt_save_dir, dst_url=os.path.join(args.train_url, "ckpt_"))


if __name__ == '__main__':
    main()
