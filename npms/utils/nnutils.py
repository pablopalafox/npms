import os
import torch
import numpy as np

def assert_models(
    shape_decoder, freeze_shape_decoder,
    pose_decoder, freeze_pose_decoder,
    shape_codes, freeze_shape_codes,
    pose_codes, freeze_pose_codes
):
    print()

    for p in shape_decoder.parameters():
        if freeze_shape_decoder:
            assert not p.requires_grad
        else:
            assert p.requires_grad

    if freeze_shape_decoder:
        print("Frozen shape decoder")
    else:
        print("UNfrozen shape decoder")


    for p in pose_decoder.parameters():
        if freeze_pose_decoder:
            assert not p.requires_grad
        else:
            assert p.requires_grad
    
    if freeze_pose_decoder:
        print("Frozen pose decoder")
    else:
        print("UNfrozen pose decoder")


    if freeze_shape_codes:
        assert not shape_codes.requires_grad
        print("Frozen shape codes")
    else:
        assert shape_codes.requires_grad
        print("UNfrozen shape codes")


    if freeze_pose_codes:
        assert not pose_codes.requires_grad
        print("Frozen pose codes")
    else:
        assert pose_codes.requires_grad
        print("UNfrozen pose codes")
    
    print()


def load_checkpoint(exp_path, checkpoint):
    # Checkpoint
    if isinstance(checkpoint, int):
        checkpoint_path = os.path.join(exp_path, "checkpoints", f"checkpoint_epoch_{checkpoint}.tar")
    else:
        checkpoint_path = os.path.join(exp_path, "checkpoints", f"{checkpoint}_checkpoint.tar")

    return torch.load(checkpoint_path)


def print_num_parameters(model):
    n_all_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print(f"Number of parameters in {model.__class__.__name__}:  {n_trainable_params} / {n_all_params}")


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, specs):
        print(specs)
        self.initial = specs['initial']
        self.interval = specs['interval']
        self.factor = specs['factor']

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)


def get_learning_rates(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(str(param_group['lr']))
    return lr_list


def print_learning_rates(optimizer):
    lr_txt = ""
    for param_group in optimizer.param_groups:
        lr_txt += " | " + str(param_group['lr'])
    print(lr_txt)


class ResBlock(torch.nn.Module):
    def __init__(self, n_out, kernel=3, normalization=torch.nn.BatchNorm3d, activation=torch.nn.ReLU):
        super().__init__()
        self.block0 = torch.nn.Sequential(
            torch.nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        )
        
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
        )

        self.block2 = torch.nn.ReLU()

    def forward(self, x0):
        x = self.block0(x0)

        x = self.block1(x)
        
        x = self.block2(x + x0)
        return x


def make_conv(n_in, n_out, n_blocks, kernel=3, stride=1, normalization=torch.nn.BatchNorm3d, activation=torch.nn.ReLU):
    blocks = []
    for i in range(n_blocks):                                                                                                                                                                                                                                                                                             
        in1 = n_in if i == 0 else n_out
        blocks.append(torch.nn.Sequential(
            torch.nn.Conv3d(in1, n_out, kernel_size=kernel, stride=stride, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        ))
    return torch.nn.Sequential(*blocks)