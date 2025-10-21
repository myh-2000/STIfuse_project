import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    '''
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    '''
    def _acquire_device(self):
        if self.args.use_gpu:
            # 不设置 CUDA_VISIBLE_DEVICES，直接访问物理设备
            physical_device_id = self.args.gpu  # 直接使用物理设备编号（如1）
        
            # 验证设备是否存在
            if physical_device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU {physical_device_id} 不存在！"
                    f"可用设备: 0-{torch.cuda.device_count()-1}"
                )
        
            device = torch.device(f'cuda:{physical_device_id}')  # 直接指向物理设备1
            print(f'Use GPU: cuda:{physical_device_id}（物理设备{physical_device_id}）')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
