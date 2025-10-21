from exp.exp_basic import Exp_Basic
from models import Autoformer, Transformer, DLinear, TimesNet, PatchTST, STIfuse, ModernTCN, Crossformer
from utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from sklearn.metrics import precision_recall_curve, auc
import os
import time
import numpy as np
import pandas as pd

class LinearClassifier(nn.Module):
    def __init__(self, 
                 feat_dim=320,   # 输入维度=16×20=320
                 num_classes=1, 
                 expansion=2,    # 特征扩展倍数（原为4，现降低）
                 dropout=0.3,
                 device=None):
        super().__init__()
        
        # 基础参数
        self.expanded_dim = expansion * feat_dim
        
        # 网络结构
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, self.expanded_dim, device=device),
            nn.LayerNorm(self.expanded_dim, device=device),
            nn.GELU()
        )
        
        # 残差块模块化
        self.res_blocks = nn.ModuleList([
            self._make_res_block(self.expanded_dim, dropout, device)
            for _ in range(3)  # 使用2个残差块代替原3个FC层
        ])
        
        # 输出层（移除偏置以降低过拟合风险）
        self.classifier = nn.Linear(self.expanded_dim, num_classes, device=device)
        
        # 初始化
        self._init_weights()

    def _make_res_block(self, dim, dropout, device):
        """构建标准残差块"""
        return nn.Sequential(
            nn.LayerNorm(dim, device=device),
            nn.Linear(dim, dim, bias=True, device=device),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True, device=device),
            nn.Dropout(dropout)
        )

    def _init_weights(self):
        """He初始化提升收敛速度"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)  # [B, 320] => [B, 640]
        
        # 残差处理
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # 标准残差连接
        
        # 分类输出（不应用Sigmoid，留给损失函数）
        return self.classifier(x).squeeze(1)  # [B, 1] => [B]

def data_pAid(critical_path, stable_path, critical_id, stable_id):
    def flat(l):
    # [[a, b], [c, d, e]] -> [a, b, c, d, e]
        new_list = []
        for i in range(len(l)):
            if isinstance(l[i], list):
                for j in range(len(l[i])):
                    new_list.append(l[i][j])
            else:
                new_list.append(l[i])
        return new_list
    
    def split_dataset(id_list, matrix_list):
        train_data = []
        val_data = []
        test_data = []

        for patient_id, matrix in zip(id_list, matrix_list):
            last_digit = int(str(patient_id)[-1])
            if 0 <= last_digit <= 5:
                train_data.append(matrix)
            elif 6 <= last_digit <= 8:
                val_data.append(matrix)
            elif last_digit == 9:
                test_data.append(matrix)
    
        return train_data, val_data, test_data

    critical = torch.load(critical_path)
    stable = torch.load(stable_path)
    critical_id = torch.load(critical_id)
    stable_id = torch.load(stable_id)
    critical_train, critical_vali, critical_test = split_dataset(critical_id, critical)
    stable_train, stable_vali, stable_test = split_dataset(stable_id, stable)
    critical_train = flat(critical_train)
    stable_train = flat(stable_train)
    critical_vali = flat(critical_vali)
    stable_vali = flat(stable_vali)
    return critical_train, stable_train, critical_vali, stable_vali, critical_test, stable_test

def data_pid(de_path, oth_path, de_id, oth_id):
    def split_dataset(id_list, matrix_list):
        # 初始化三个空集合
        train_data = []
        val_data = []
        test_data = []
    
        # 同时遍历ID和对应的矩阵
        for patient_id, matrix in zip(id_list, matrix_list):
        # 提取尾数
            last_digit = int(str(patient_id)[-1])
        
            # 根据尾数分配数据
            if 0 <= last_digit <= 5:
                train_data.append(patient_id)
            elif 6 <= last_digit <= 8:
                val_data.append(patient_id)
            elif last_digit == 9:
                test_data.append(patient_id)
    
        return train_data, val_data, test_data

    de = torch.load(de_id)
    oth = torch.load(oth_id)
    de_id = torch.load(de_id)
    oth_id = torch.load(oth_id)
    _, _, de_patient_test = split_dataset(de_id, de)
    _, _, oth_patient_test = split_dataset(oth_id, oth)

    return de_patient_test, oth_patient_test

def new_loss_was(enc_out, sti_out, m, n, tem=0.1):
    (batch_size, a) = enc_out.shape
    mu_matrix = enc_out
    sigma_matrix = sti_out
    mu_d_matrix = torch.norm((mu_matrix.unsqueeze(0) - mu_matrix.unsqueeze(1)), dim=2)
    sigma_d_matrix = torch.norm((sigma_matrix.unsqueeze(0) - sigma_matrix.unsqueeze(1)), dim=2)
    d = mu_d_matrix + sigma_d_matrix
    d = d + torch.eye(batch_size, device=enc_out.device)#为了后续开根号的时候求梯度不会爆炸
    d = torch.sqrt(d / a)
    matrix_S = 1 - d + torch.eye(batch_size, device=enc_out.device)
    matrix_S = matrix_S / tem
    matrix_S = torch.exp(matrix_S)
    S_cri = matrix_S[:m]
    diag_val = matrix_S[:m, :m].diag()
    S_sta1 = matrix_S[-n:, :m]
    S_sta2 = matrix_S[-n:, -n:]

    #L_critical
    L_critical = diag_val / torch.sum(S_cri, dim=1)
    L_critical = -torch.log(L_critical)

    #L_stable
    L2 = torch.sum(S_sta1, dim=1) / S_sta2.transpose(0,1)
    L2 = L2.transpose(0,1)
    L1 = L2 + torch.ones_like(S_sta2)
    L1 = 1 / L1
    L1 = -torch.log(L1)
    L_stable = L1 - L1.diag() * torch.eye(n, device=matrix_S.device)
    Loss = 0.1*torch.sum(L_critical) / m + torch.sum(L_stable) / (n * n - n)
    return Loss
    
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self._load_all_data()

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'PatchTST': PatchTST,
            'STIfuse': STIfuse,
            'ModernTCN': ModernTCN,
            'Crossformer': Crossformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _load_all_data(self):
        """一次性加载所有数据并存储在类属性中"""
        # 加载主要数据
        (self.train_data_critical, self.train_data_stable, 
        self.vali_data_critical, self.vali_data_stable, 
        self.test_data_critical, self.test_data_stable) = data_pAid(
            self.args.critical_path, self.args.stable_path, 
            self.args.critical_id_path, self.args.stable_id_path)
    
        # 加载刺激数据
        (self.train_sti_critical, self.train_sti_stable, 
        self.vali_sti_critical, self.vali_sti_stable, 
        self.test_sti_critical, self.test_sti_stable) = data_pAid(
            self.args.critical_sti_path, self.args.stable_sti_path, 
            self.args.critical_id_path, self.args.stable_id_path)

        # 计算数据长度
        self.train_num_critical = len(self.train_data_critical)
        self.train_num_stable = len(self.train_data_stable)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        print(self.device)
        m = self.args.m
        n = self.args.n
        pos_weight = torch.tensor([m/n]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        return criterion
    
    def train_classifier(self, setting):
        train_data_critical = self.train_data_critical
        train_data_stable = self.train_data_stable
        vali_data_critical = self.vali_data_critical
        vali_data_stable = self.vali_data_stable
        test_data_critical = self.test_data_critical
        test_data_stable = self.test_data_stable
        
        train_num_critical = self.train_num_critical
        train_num_stable = self.train_num_stable
        
        print(train_num_critical, train_num_stable)
        path = os.path.join(self.args.checkpoints, setting)

        def vali_classifier2(train_data_critical, train_data_stable, loss1, model):
            batch_size = 256
            m = 128
            n = 128
            batch_label = torch.cat([torch.zeros(m), torch.ones(n)])
            batch_label = batch_label.float().to(self.device)
            train_num_stable = len(train_data_stable)
            train_num_critical = len(train_data_critical)
            model.eval()
            train_loss = []
            train_num_critical = len(train_data_critical)
            train_num_stable = len(train_data_stable)
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            for j in range(int(train_num_stable*1/batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                batch_x = batch_x.float().to(self.device)
                with torch.no_grad():
                    outputs = self.model(batch_x)
                loss = loss1(outputs, batch_label)
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            print(train_loss)        
            return train_loss

        batch_size = self.args.batch_size
        m = self.args.m
        n = self.args.n
        wlen = self.args.wlen
        vlen = self.args.vlen  
        
        if not os.path.exists(path):
            os.makedirs(path)
        nvar = self.args.nvar

        train_steps = int(train_num_stable*1/batch_size)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        self.model.train()
        #####################训练判别器####################################################
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        train_steps = int(train_num_stable*1/batch_size)
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        batch_label = torch.cat([torch.zeros(m), torch.ones(n)])
        batch_label = batch_label.float().to(self.device)
        for epoch in range(self.args.train_epochs):
            train_loss = []
            epoch_time = time.time()
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            for j in range(int(train_num_stable*1/batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])
                
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_label)
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)        
            print(epoch + 1, train_steps, train_loss)

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            vali_loss = vali_classifier2(vali_data_critical, vali_data_stable, criterion, self.model)
            if np.isnan(vali_loss):
                print('np.isnan(train_loss)', np.isnan(vali_loss))
                break
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
###################测试################################################            
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():  
            patient_score_critical = []
            patient_score_stable = []
            thresholds = torch.tensor([99]).float().to(self.device)

            for i in range(len(test_data_critical)): 
                if test_data_critical[i]:
                    batch = torch.stack(test_data_critical[i])              
                    batch = batch.float().to(self.device)
                    test_out1 = self.model(batch)                    
                    thresholds = torch.cat([thresholds, test_out1])
                    patient_score_critical.append(test_out1)
            for i in range(len(test_data_stable)):
                if test_data_stable[i]:
                    batch = torch.stack(test_data_stable[i])
                    batch = batch.float().to(self.device)
                    test_out1 = self.model(batch)
                    thresholds = torch.cat([thresholds, test_out1])
                    patient_score_stable.append(test_out1)
            thresholds, _ = torch.sort(thresholds)
            thresholds = torch.round(thresholds * 1000) / 1000
            thresholds = torch.unique(thresholds)            
            ReCall = np.zeros(len(thresholds))
            Precision = np.zeros(len(thresholds))
            FPR = np.zeros(len(thresholds))
            sensitivity = np.zeros(len(thresholds))
            for i, t in enumerate(thresholds):
                alarm_num = 0.000001
                true_alarm = 0
                c_event = 0
                FP = 0
                no_event = 0
                event_num = len(patient_score_critical)
                TP = 0
                all_event = 0
                for j in range(len(patient_score_critical)):
                    has_element_greater_than_thresholds = (patient_score_critical[j] <= t).any()
                    num_greater_than_thresholds = torch.sum(patient_score_critical[j] <= t).item()
                    true_alarm = true_alarm + num_greater_than_thresholds
                    alarm_num = alarm_num + num_greater_than_thresholds
                    TP = TP + num_greater_than_thresholds
                    all_event = all_event + len(patient_score_critical[j])
                    if has_element_greater_than_thresholds:
                        c_event = c_event + 1
                for k in range(len(patient_score_stable)):
                    num_greater_than_thresholds = torch.sum(patient_score_stable[k] <= t).item()
                    alarm_num = alarm_num + num_greater_than_thresholds
                    FP = FP + num_greater_than_thresholds
                    no_event = no_event + len(patient_score_stable[k])
                ReCall[i] = c_event / event_num
                Precision[i] = true_alarm / alarm_num 
                FPR[i] = FP / no_event
                sensitivity[i] = TP / all_event
            df = pd.DataFrame({'x': ReCall, 'y': Precision})
            result_df = df.groupby('x', as_index=False)['y'].max()

            ReCall = result_df['x'].to_numpy()
            Precision = result_df['y'].to_numpy()
            auprc = auc(ReCall, Precision)
            df = pd.DataFrame({'x': FPR, 'y': sensitivity})
            result_df = df.groupby('x', as_index=False)['y'].max()

            FPR = result_df['x'].to_numpy()
            sensitivity = result_df['y'].to_numpy()

            auroc = auc(FPR, sensitivity)
            print("auprc:", auprc, "auroc:", auroc)
        return auprc

    def pretrain_STIfuse(self, setting):
        train_data_critical = self.train_data_critical
        train_data_stable = self.train_data_stable
        vali_data_critical = self.vali_data_critical
        vali_data_stable = self.vali_data_stable

        train_sti_critical = self.train_sti_critical
        train_sti_stable = self.train_sti_stable
        vali_sti_critical = self.vali_sti_critical
        vali_sti_stable = self.vali_sti_stable
        
        train_num_critical = self.train_num_critical
        train_num_stable = self.train_num_stable

        train_num_critical = len(train_data_critical)
        train_num_stable = len(train_data_stable)
        batch_size = self.args.batch_size
        m = self.args.m
        n = self.args.n
        wlen = self.args.wlen
        vlen = self.args.vlen 
        nvar = self.args.nvar
        print(train_num_critical, train_num_stable)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_data_critical)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        def vali(train_data_critical, train_sti_critical, train_data_stable, train_sti_stable, loss1, model):
            batch_size = 256
            m = 128
            n = 128
            train_num_stable = len(train_data_stable)
            train_num_critical = len(train_data_critical)
            model.eval()
            train_loss = []
            train_num_critical = len(train_data_critical)
            train_num_stable = len(train_data_stable)
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            train_sti_critical = [train_sti_critical[i] for i in critical_random]
            train_sti_stable = [train_sti_stable[i] for i in stable_random]
            for j in range(int(train_num_stable*1/batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])
                batch_y = torch.zeros([batch_size, vlen])
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                        batch_y[i] = train_sti_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                        batch_y[i] = train_sti_stable[(i - m + j * n) % train_num_stable ]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                with torch.no_grad():
                    enc_out, mlp_out = model(batch_x, batch_y)
                    loss = new_loss_was(enc_out, mlp_out, m, n, tem=self.args.tem)
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            return train_loss

        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        batch_size = self.args.batch_size
        m = self.args.m
        n = self.args.n
        wlen = self.args.wlen
        vlen = self.args.vlen
        loss1 = new_loss_was
        for epoch in range(self.args.train_epochs):
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            train_sti_critical = [train_sti_critical[i] for i in critical_random]
            train_sti_stable = [train_sti_stable[i] for i in stable_random]
            train_loss = []
            id = 0
            self.model.train()
            epoch_time = time.time()
            for j in range(int(train_num_stable*1/batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])

                batch_y = torch.zeros([batch_size, vlen])
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                        batch_y[i] = train_sti_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                        batch_y[i] = train_sti_stable[(i - m + j * n) % train_num_stable ]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                model_optim.zero_grad()
                enc_out, mlp_out = self.model(batch_x, batch_y)
                loss = new_loss_was(enc_out, mlp_out, m, n, tem=self.args.tem)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print(epoch + 1, train_steps, train_loss)
            vali_loss = vali(vali_data_critical, vali_sti_critical, vali_data_stable, vali_sti_stable, loss1, self.model)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return self.model
    
    def clinical_viability_classifier(self, setting):
        train_data_critical = self.train_data_critical
        train_data_stable = self.train_data_stable
        vali_data_critical = self.vali_data_critical
        vali_data_stable = self.vali_data_stable

        test_data_critical = self.test_data_critical
        test_data_stable = self.test_data_stable

        train_sti_critical = self.train_sti_critical
        train_sti_stable = self.train_sti_stable
        vali_sti_critical = self.vali_sti_critical
        vali_sti_stable = self.vali_sti_stable

        test_sti_critical = self.test_sti_critical
        test_sti_stable = self.test_sti_stable
        
        train_num_critical = self.train_num_critical
        train_num_stable = self.train_num_stable

        train_num_critical = len(train_data_critical)
        train_num_stable = len(train_data_stable)
        vali_num_critical = len(vali_data_critical)
        vali_num_stable = len(vali_data_stable)
        path = os.path.join(self.args.checkpoints, setting)
        path_cla = path + '_classification'

        def vali_classifier1(train_enc_critical, train_mlp_critical, train_enc_stable, train_mlp_stable, loss1, classifier):
            batch_size = 256
            m = 128
            n = 128
            batch_label = torch.cat([torch.zeros(m), torch.ones(n)])
            batch_label = batch_label.float().to(self.device)
            train_num_stable = len(train_enc_stable)
            train_num_critical = len(train_enc_critical)
            
            classifier.eval()
            train_loss = []
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_enc_critical = [train_enc_critical[i] for i in critical_random]
            train_enc_stable = [train_enc_stable[i] for i in stable_random]
            train_mlp_critical = [train_mlp_critical[i] for i in critical_random]
            train_mlp_stable = [train_mlp_stable[i] for i in stable_random]
            for j in range(int(train_num_stable*1/batch_size)):
                batch_enc = torch.zeros([batch_size, 16])
                batch_mlp = torch.zeros([batch_size, 16])
                for i in range(batch_size):
                    if i < m:
                        batch_enc[i] = train_enc_critical[(i + j * m) % train_num_critical ]
                        batch_mlp[i] = train_mlp_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_enc[i] = train_enc_stable[(i - m + j * n) % train_num_stable ]
                        batch_mlp[i] = train_mlp_stable[(i - m + j * n) % train_num_stable ]
                model_optim.zero_grad()
                batch_enc = batch_enc.float().to(self.device)
                batch_mlp = batch_mlp.float().to(self.device)
                with torch.no_grad():
                    inputs = torch.cat([batch_enc, batch_mlp], dim=1).float().to(self.device)
                    outputs = classifier(inputs)
                loss = loss1(outputs, batch_label)
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            print(train_loss)        
            return train_loss
    
        batch_size = self.args.batch_size
        m = self.args.m
        n = self.args.n
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_cla):
            os.makedirs(path_cla)
        classifier = LinearClassifier(feat_dim=16*2, num_classes=1, device=self.args.gpu)

        train_steps = int(train_num_stable*1/batch_size)
        early_stopping1 = EarlyStopping(patience=15, verbose=True)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        self.model.eval()

#####################训练判别器####################################################
        classifier.train()
        model_optim = optim.Adam(classifier.parameters(), lr=0.001)
        train_steps = int(train_num_stable*1/batch_size)
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = 0.001)
        batch_label = torch.cat([torch.zeros(m), torch.ones(n)])
        batch_label = batch_label.float().to(self.device)

        with torch.no_grad():
            num = 10000
            batch_x = torch.stack(train_data_critical).float().to(self.device)
            batch_y = torch.stack(train_sti_critical).float().to(self.device)
            for i in range(int(train_num_critical / num) + 1):
                border1 = i * num
                border2 = min((i + 1) * num, train_num_critical)
                enc_out1, mlp_out1 = self.model(batch_x[border1:border2], batch_y[border1:border2])
                if i == 0:
                    train_enc_de = enc_out1
                    train_mlp_de = mlp_out1
                else:
                    train_enc_de = torch.cat([train_enc_de, enc_out1], dim=0)
                    train_mlp_de = torch.cat([train_mlp_de, mlp_out1], dim=0)
            train_enc_de = train_enc_de.cpu()
            train_mlp_de = train_mlp_de.cpu()
            batch_x = torch.stack(train_data_stable).float().to(self.device)
            batch_y = torch.stack(train_sti_stable).float().to(self.device)
            for i in range(int(train_num_stable / num) + 1):
                print(i)
                border1 = i * num
                border2 = min((i + 1) * num, train_num_stable)
                enc_out1, mlp_out1 = self.model(batch_x[border1:border2], batch_y[border1:border2])
                if i == 0:
                    train_enc_stable = enc_out1
                    train_mlp_stable = mlp_out1
                else:
                    train_enc_stable = torch.cat([train_enc_stable, enc_out1], dim=0)
                    train_mlp_stable = torch.cat([train_mlp_stable, mlp_out1], dim=0) 
            train_enc_stable = train_enc_stable.cpu()
            train_mlp_stable = train_mlp_stable.cpu()
            batch_x = torch.stack(vali_data_critical).float().to(self.device)
            batch_y = torch.stack(vali_sti_critical).float().to(self.device)
            for i in range(int(vali_num_critical / num) + 1):
                print(i)
                border1 = i * num
                border2 = min((i + 1) * num, vali_num_critical)
                enc_out1, mlp_out1 = self.model(batch_x[border1:border2], batch_y[border1:border2])
                if i == 0:
                    vali_enc_critical = enc_out1
                    vali_mlp_critical = mlp_out1
                else:
                    vali_enc_critical = torch.cat([vali_enc_critical, enc_out1], dim=0)
                    vali_mlp_critical = torch.cat([vali_mlp_critical, mlp_out1], dim=0)
            vali_enc_critical = vali_enc_critical.cpu()
            vali_mlp_critical = vali_mlp_critical.cpu()
            batch_x = torch.stack(vali_data_stable).float().to(self.device)
            batch_y = torch.stack(vali_sti_stable).float().to(self.device)
            for i in range(int(vali_num_stable / num) + 1):    
                print(i)
                border1 = i * num
                border2 = min((i + 1) * num, vali_num_stable)
                enc_out1, mlp_out1 = self.model(batch_x[border1:border2], batch_y[border1:border2])
                if i == 0:
                    vali_enc_stable = enc_out1
                    vali_mlp_stable = mlp_out1
                else:
                    vali_enc_stableh = torch.cat([vali_enc_stable, enc_out1], dim=0)
                    vali_mlp_stable = torch.cat([vali_mlp_stable, mlp_out1], dim=0)
            vali_enc_stable = vali_enc_stable.cpu()
            vali_mlp_stable = vali_mlp_stable.cpu()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            epoch_time = time.time()
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)

            train_enc_critical = [train_enc_critical[i] for i in critical_random]
            train_enc_stable = [train_enc_stable[i] for i in stable_random]
            train_mlp_critical = [train_mlp_critical[i] for i in critical_random]
            train_mlp_stable = [train_mlp_stable[i] for i in stable_random]
            for j in range(int(train_num_stable * 1 / batch_size)):

                batch_enc = torch.zeros([batch_size, 16])
                batch_mlp = torch.zeros([batch_size, 16])

                for i in range(batch_size):
                    if i < m:
                        batch_enc[i] = train_enc_critical[(i + j * m) % train_num_critical ]
                        batch_mlp[i] = train_mlp_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_enc[i] = train_enc_stable[(i - m + j * n) % train_num_stable ]
                        batch_mlp[i] = train_mlp_stable[(i - m + j * n) % train_num_stable ]
                model_optim.zero_grad()
                batch_enc = batch_enc.float().to(self.device)
                batch_mlp = batch_mlp.float().to(self.device)
                inputs = torch.cat([batch_enc, batch_mlp], dim=1)
                outputs = classifier(inputs.detach())
                loss = criterion(outputs, batch_label)
                loss.backward()
                model_optim.step()
                scheduler.step()
                train_loss.append(loss.item())
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)        
            print(epoch + 1, train_steps, train_loss)

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            vali_loss = vali_classifier1(vali_enc_critical, vali_mlp_critical, vali_enc_stable, vali_mlp_stable, criterion, classifier)
            if np.isnan(vali_loss):
                print('np.isnan(train_loss)', np.isnan(vali_loss))
                break
            early_stopping1(vali_loss, classifier, path_cla)
            if early_stopping1.early_stop:
                print("Early stopping")
                break

        classifier.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '_classification', 'checkpoint.pth')))
        classifier.eval()
        with torch.no_grad():
            patient_score_critical = []
            patient_score_stable = []
            thresholds = torch.tensor([99]).float().to(self.device)
           
            for i in range(len(test_data_critical)):
                if test_data_critical[i]:
                    batch = torch.stack(test_data_critical[i])
                    batch_sti = torch.stack(test_sti_critical[i])
                    batch = batch.float().to(self.device)
                    batch_sti = batch_sti.float().to(self.device)
                    enc_out1, mlp_out1 = self.model(batch, batch_sti)
                    inputs = torch.cat([enc_out1, mlp_out1], dim=1).float().to(self.device)
                    test_out1 = classifier(inputs)
                    thresholds = torch.cat([thresholds, test_out1])
                    patient_score_critical.append(test_out1)
            
            for i in range(len(test_data_stable)):
                if test_data_stable[i]:
                    batch = torch.stack(test_data_stable[i])
                    batch_sti = torch.stack(test_sti_stable[i])
                    batch = batch.float().to(self.device)
                    batch_sti = batch_sti.float().to(self.device)
                    enc_out1, mlp_out1 = self.model(batch, batch_sti)
                    inputs = torch.cat([enc_out1, mlp_out1], dim=1).float().to(self.device)
                    test_out1 = classifier(inputs)
                    thresholds = torch.cat([thresholds, test_out1])
                    patient_score_stable.append(test_out1)

            thresholds, _ = torch.sort(thresholds)
            thresholds = torch.round(thresholds * 1000) / 1000
            thresholds = torch.unique(thresholds)
            ReCall = np.zeros(len(thresholds))
            Precision = np.zeros(len(thresholds))
            FPR = np.zeros(len(thresholds))
            sensitivity = np.zeros(len(thresholds))
            for i, t in enumerate(thresholds):
                alarm_num = 0.000001
                true_alarm = 0
                c_event = 0
                FP = 0
                no_event = 0
                event_num = len(patient_score_critical)
                TP = 0
                all_event = 0
                for j in range(len(patient_score_critical)):
                    has_element_greater_than_thresholds = (patient_score_critical[j] <= t).any()
                    num_greater_than_thresholds = torch.sum(patient_score_critical[j] <= t).item()
                    true_alarm = true_alarm + num_greater_than_thresholds
                    alarm_num = alarm_num + num_greater_than_thresholds
                    TP = TP + num_greater_than_thresholds
                    all_event = all_event + len(patient_score_critical[j])
                    if has_element_greater_than_thresholds:
                        c_event = c_event + 1
                for k in range(len(patient_score_stable)):
                    num_greater_than_thresholds = torch.sum(patient_score_stable[k] <= t).item()
                    alarm_num = alarm_num + num_greater_than_thresholds
                    FP = FP + num_greater_than_thresholds
                    no_event = no_event + len(patient_score_stable[k])
                ReCall[i] = c_event / event_num
                Precision[i] = true_alarm / alarm_num 
                FPR[i] = FP / no_event
                sensitivity[i] = TP / all_event
            ###画PR曲线
            ReCall1 = sensitivity
            df = pd.DataFrame({'x': ReCall, 'y': Precision})
            result_df = df.groupby('x', as_index=False)['y'].max()
            # 转换为numpy数组
            ReCall = result_df['x'].to_numpy()
            Precision = result_df['y'].to_numpy()
            auprc = auc(ReCall, Precision)
            ###画ROC曲线
            df = pd.DataFrame({'x': FPR, 'y': ReCall1})
            result_df = df.groupby('x', as_index=False)['y'].max()

            # 转换为numpy数组
            FPR = result_df['x'].to_numpy()
            ReCall1 = result_df['y'].to_numpy()

            auroc = auc(FPR, ReCall1)
            print("auprc:", auprc, "auroc:", auroc)
        return auprc, ReCall, Precision, auroc, FPR, ReCall1, Precision[-1]

    def STIfuse_finetune(self, setting):
        train_data_critical = self.train_data_critical
        train_data_stable = self.train_data_stable
        vali_data_critical = self.vali_data_critical
        vali_data_stable = self.vali_data_stable

        train_sti_critical = self.train_sti_critical
        train_sti_stable = self.train_sti_stable
        vali_sti_critical = self.vali_sti_critical
        vali_sti_stable = self.vali_sti_stable
        
        train_num_critical = self.train_num_critical
        train_num_stable = self.train_num_stable

        train_num_critical = len(train_data_critical)
        train_num_stable = len(train_data_stable)
        train_num_critical = len(train_data_critical)
        train_num_stable = len(train_data_stable)
        batch_size = self.args.batch_size
        m = self.args.m
        n = self.args.n
        wlen = self.args.wlen
        vlen = self.args.vlen 
        nvar = self.args.nvar
        print(train_num_critical, train_num_stable)
        path = os.path.join(self.args.checkpoints, setting)
        path_ft = path + '_finetune'
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(path_ft):
            os.makedirs(path_ft)

        time_now = time.time()

        train_steps = len(train_data_critical)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        def vali(train_data_critical, train_sti_critical, train_data_stable, train_sti_stable, loss1, model):
            batch_size = 256
            m = 128
            n = 128
            train_num_stable = len(train_data_stable)
            train_num_critical = len(train_data_critical)
            model.eval()
            train_loss = []
            train_num_critical = len(train_data_critical)
            train_num_stable = len(train_data_stable)
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            train_sti_critical = [train_sti_critical[i] for i in critical_random]
            train_sti_stable = [train_sti_stable[i] for i in stable_random]
            for j in range(int(train_num_stable * 1 / batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])
                batch_y = torch.zeros([batch_size, vlen])
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                        batch_y[i] = train_sti_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                        batch_y[i] = train_sti_stable[(i - m + j * n) % train_num_stable ]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                with torch.no_grad():
                    enc_out, mlp_out = model(batch_x, batch_y)
                    loss, mu_cri, mu_cri_sta, mu_sta, sigma_cri, sigma_cri_sta, sigma_sta, cri, sta= new_loss_was(enc_out, mlp_out, m, n, tem=self.args.tem)
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            print(train_loss)
            print(train_loss)        
            return train_loss
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        batch_size = self.args.batch_size
        m = self.args.m
        n = self.args.n
        wlen = self.args.wlen
        input_dim = self.args.input_dim
        vlen = self.args.vlen
        loss1 = new_loss_was
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        model_optim = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate * 0.1,
            weight_decay=1e-4
        )
        # 将 OneCycleLR 参数调整为恒定
        scheduler = lr_scheduler.LambdaLR(
            optimizer=model_optim,
            lr_lambda=lambda epoch: 1.0  # 始终返回1，保持学习率不变
        )

        for epoch in range(1):
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            train_sti_critical = [train_sti_critical[i] for i in critical_random]
            train_sti_stable = [train_sti_stable[i] for i in stable_random]
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for j in range(int(train_num_stable*1/batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])
                batch_y = torch.zeros([batch_size, vlen])
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                        batch_y[i] = train_sti_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                        batch_y[i] = train_sti_stable[(i - m + j * n) % train_num_stable ]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                model_optim.zero_grad()
                enc_out, mlp_out = self.model(batch_x, batch_y)
                loss = new_loss_was(enc_out, mlp_out, m, n, tem=self.args.tem)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
                scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print(epoch + 1, train_steps, train_loss)
            vali_loss = vali(vali_data_critical, vali_sti_critical, vali_data_stable, vali_sti_stable, loss1, self.model)
            early_stopping(vali_loss, self.model, path_ft)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        return 

def clinical_viability_classifier_finutune(self, setting):
        train_data_critical = self.train_data_critical
        train_data_stable = self.train_data_stable
        vali_data_critical = self.vali_data_critical
        vali_data_stable = self.vali_data_stable

        test_data_critical = self.test_data_critical
        test_data_stable = self.test_data_stable

        train_sti_critical = self.train_sti_critical
        train_sti_stable = self.train_sti_stable
        vali_sti_critical = self.vali_sti_critical
        vali_sti_stable = self.vali_sti_stable

        test_sti_critical = self.test_sti_critical
        test_sti_stable = self.test_sti_stable
        
        train_num_critical = self.train_num_critical
        train_num_stable = self.train_num_stable

        train_num_critical = len(train_data_critical)
        train_num_stable = len(train_data_stable)
        path = os.path.join(self.args.checkpoints, setting)

        def vali_classifier1(train_data_critical, train_sti_critical, train_data_stable, train_sti_stable, loss1, model, classifier):
            batch_size = 256
            m = 128
            n = 128
            batch_label = torch.cat([torch.zeros(m), torch.ones(n)])
            batch_label = batch_label.float().to(self.device)
            train_num_stable = len(train_data_stable)
            train_num_critical = len(train_data_critical)
            model.eval()
            classifier.eval()
            train_loss = []
            train_num_critical = len(train_data_critical)
            train_num_stable = len(train_data_stable)
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            train_sti_critical = [train_sti_critical[i] for i in critical_random]
            train_sti_stable = [train_sti_stable[i] for i in stable_random]
            for j in range(int(train_num_stable*1/batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])
                batch_y = torch.zeros([batch_size, vlen])
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                        batch_y[i] = train_sti_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                        batch_y[i] = train_sti_stable[(i - m + j * n) % train_num_stable ]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                with torch.no_grad():
                    enc_out, mlp_out = model(batch_x, batch_y)
                    inputs = torch.cat([enc_out, mlp_out], dim=1)
                    outputs = classifier(inputs)
                loss = loss1(outputs, batch_label)
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            print(train_loss)        
            return train_loss

        batch_size = self.args.batch_size
        m = self.args.m
        n = self.args.n
        wlen = self.args.wlen
        vlen = self.args.vlen  
        
        if not os.path.exists(path):
            os.makedirs(path)
        nvar = self.args.nvar
        classifier = LinearClassifier(feat_dim=16 * 2, num_classes=1, device=self.args.gpu)
        classifier.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '_classification', 'checkpoint.pth')))

        model_optim = optim.Adam(classifier.parameters(),
            lr=self.args.learning_rate * 0.1,
            weight_decay=1e-4
        )
        train_steps = int(train_num_stable*1/batch_size)
        # 将 OneCycleLR 参数调整为恒定
        scheduler = lr_scheduler.LambdaLR(
            optimizer=model_optim,
            lr_lambda=lambda epoch: 1.0  # 始终返回1，保持学习率不变
        )
        
        early_stopping1 = EarlyStopping(patience=5, verbose=True)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '_finetune', 'checkpoint.pth')))
        self.model.eval()
        path_cla = path + '_classification'
        path_cla_ft = path_cla + '_finetune'
        if not os.path.exists(path_cla_ft):
            os.makedirs(path_cla_ft)

#####################训练判别器####################################################
        classifier.train()
        train_steps = int(train_num_stable*1/batch_size)
        batch_label = torch.cat([torch.zeros(m), torch.ones(n)])
        batch_label = batch_label.float().to(self.device)
        for epoch in range(1):
            train_loss = []
            epoch_time = time.time()
            critical_random = torch.randperm(train_num_critical)
            stable_random = torch.randperm(train_num_stable)
            train_data_critical = [train_data_critical[i] for i in critical_random]
            train_data_stable = [train_data_stable[i] for i in stable_random]
            train_sti_critical = [train_sti_critical[i] for i in critical_random]
            train_sti_stable = [train_sti_stable[i] for i in stable_random]
            train_time_critical = [train_time_critical[i] for i in critical_random]
            train_time_stable = [train_time_stable[i] for i in stable_random]
            for j in range(int(train_num_stable*1/batch_size)):
                batch_x = torch.zeros([batch_size, wlen, nvar])
                batch_y = torch.zeros([batch_size, vlen])
                for i in range(batch_size):
                    if i < m:
                        batch_x[i] = train_data_critical[(i + j * m) % train_num_critical ]
                        batch_y[i] = train_sti_critical[(i + j * m) % train_num_critical ]
                    else:
                        batch_x[i] = train_data_stable[(i - m + j * n) % train_num_stable ]
                        batch_y[i] = train_sti_stable[(i - m + j * n) % train_num_stable ]
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                with torch.no_grad():
                    enc_out, mlp_out = self.model(batch_x, batch_y)
    
                inputs = torch.cat([enc_out, mlp_out], dim=1)
                outputs = classifier(inputs.detach())
                #print(outputs)
                loss = criterion(outputs, batch_label)

                #print(loss)
                loss.backward()
                model_optim.step()
                scheduler.step()
                #id = id + 1
                train_loss.append(loss.item())
            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            #print("TPR", TPR/id, "TNR", FPR/id, "Rate", Rate/id)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)        
            print(epoch + 1, train_steps, train_loss)
            #early_stopping(train_loss, self.model, path)

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            vali_loss = vali_classifier1(vali_data_critical, vali_sti_critical, vali_data_stable, vali_sti_stable, criterion, self.model, classifier)
            if np.isnan(vali_loss):
                print('np.isnan(train_loss)', np.isnan(vali_loss))
                break
            early_stopping1(vali_loss, classifier, path_cla_ft)
            if early_stopping1.early_stop:
                print("Early stopping")
                break

###################测试################################################

        classifier.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '_classification' + '_finetune', 'checkpoint.pth')))
        #test
        classifier.eval()
        with torch.no_grad():
            patient_score_critical = []
            patient_score_stable = []
            thresholds = torch.tensor([99]).float().to(self.device)
           
            for i in range(len(test_data_critical)):
                
                if test_data_critical[i]:
                    batch = torch.stack(test_data_critical[i])
                    batch_sti = torch.stack(test_sti_critical[i])
                    batch = batch.float().to(self.device)
                    batch_sti = batch_sti.float().to(self.device)
                    enc_out1, mlp_out1 = self.model(batch, batch_sti)
                    inputs = torch.cat([enc_out1, mlp_out1], dim=1).float().to(self.device)
                    test_out1 = classifier(inputs)
                    thresholds = torch.cat([thresholds, test_out1])
                    patient_score_critical.append(test_out1)
            
            for i in range(len(test_data_stable)):
                if test_data_stable[i]:
                    batch = torch.stack(test_data_stable[i])
                    batch_sti = torch.stack(test_sti_stable[i])
                    batch = batch.float().to(self.device)
                    batch_sti = batch_sti.float().to(self.device)
                    enc_out1, mlp_out1 = self.model(batch, batch_sti)
                    inputs = torch.cat([enc_out1, mlp_out1], dim=1).float().to(self.device)
                    test_out1 = classifier(inputs)
                    thresholds = torch.cat([thresholds, test_out1])
                    patient_score_stable.append(test_out1)
            
           
            thresholds, _ = torch.sort(thresholds)
            thresholds = torch.round(thresholds * 1000) / 1000
            thresholds = torch.unique(thresholds)
            ReCall = np.zeros(len(thresholds))
            Precision = np.zeros(len(thresholds))
            FPR = np.zeros(len(thresholds))
            sensitivity = np.zeros(len(thresholds))
            for i, t in enumerate(thresholds):
                alarm_num = 0.000001
                true_alarm = 0
                c_event = 0
                FP = 0
                no_event = 0
                event_num = len(patient_score_critical)
                TP = 0
                all_event = 0
                for j in range(len(patient_score_critical)):
                    has_element_greater_than_thresholds = (patient_score_critical[j] <= t).any()
                    num_greater_than_thresholds = torch.sum(patient_score_critical[j] <= t).item()
                    true_alarm = true_alarm + num_greater_than_thresholds
                    alarm_num = alarm_num + num_greater_than_thresholds
                    TP = TP + num_greater_than_thresholds
                    all_event = all_event + len(patient_score_critical[j])
                    if has_element_greater_than_thresholds:
                        c_event = c_event + 1
                for k in range(len(patient_score_stable)):
                    num_greater_than_thresholds = torch.sum(patient_score_stable[k] <= t).item()
                    alarm_num = alarm_num + num_greater_than_thresholds
                    FP = FP + num_greater_than_thresholds
                    no_event = no_event + len(patient_score_stable[k])
                ReCall[i] = c_event / event_num
                Precision[i] = true_alarm / alarm_num 
                FPR[i] = FP / no_event
                sensitivity[i] = TP / all_event
            ###画PR曲线
            ReCall1 = sensitivity
            df = pd.DataFrame({'x': ReCall, 'y': Precision})
            result_df = df.groupby('x', as_index=False)['y'].max()
            # 转换为numpy数组
            ReCall = result_df['x'].to_numpy()
            Precision = result_df['y'].to_numpy()           
            auprc = auc(ReCall, Precision)
            df = pd.DataFrame({'x': FPR, 'y': ReCall1})
            result_df = df.groupby('x', as_index=False)['y'].max()
            FPR = result_df['x'].to_numpy()
            ReCall1 = result_df['y'].to_numpy()
            auroc = auc(FPR, ReCall1)
            print("auprc:", auprc, "auroc:", auroc)
        return auprc
    
    



    