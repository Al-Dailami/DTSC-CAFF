import torch
import torch.nn as nn
from torch import cat, exp
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.modules.batchnorm import _BatchNorm


class my_AFF(nn.Module):
    '''
    Point-wise Convolution based Attention module (PWAtt)
    '''

    def __init__(self, channels=64, r=2):
        super(my_AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=inter_channels, out_channels=channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xa = self.local_att(x)
        wei = self.sigmoid(xa)

        xo = 2 * x * wei
        return xo, wei

# Root Mean Squared Logarithmic Error (RMSLE) loss
class RMSLELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSLELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the log(predictions) corresponding to no data should be set to 0
        # log_y_hat = y_hat.log().where(mask, torch.zeros_like(y))
        log_y_hat = torch.log(y_hat + 1).where(mask, torch.zeros_like(y))
        # the we set the log(labels) that correspond to no data to be 0 as well
        # log_y = y.log().where(mask, torch.zeros_like(y))
        log_y = torch.log(y + 1).where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        rmsle_loss = torch.sqrt(loss + self.eps)
        loss = torch.sum(rmsle_loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()

# Root Mean Squared Error (MSE) loss
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the predictions corresponding to no data should be set to 0
        y_hat = y_hat.where(mask, torch.zeros_like(y))
        # the we set the labels that correspond to no data to be 0 as well
        y = y.where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(y_hat, y)
        rmse_loss = torch.sqrt(loss + self.eps)
        loss = torch.sum(rmse_loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()

# Mean Squared Logarithmic Error (MSLE) loss
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the log(predictions) corresponding to no data should be set to 0
        log_y_hat = y_hat.log().where(mask, torch.zeros_like(y))
        # the we set the log(labels) that correspond to no data to be 0 as well
        log_y = y.log().where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


# Mean Squared Error (MSE) loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the predictions corresponding to no data should be set to 0
        y_hat = y_hat.where(mask, torch.zeros_like(y))
        # the we set the labels that correspond to no data to be 0 as well
        y = y.where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(y_hat, y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


class MyBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MyBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # hack to work around model.eval() issue
        if not self.training:
            self.eval_momentum = 0  # set the momentum to zero when the model is validating

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum if self.training else self.eval_momentum

        if self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum if self.training else self.eval_momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            training=True, momentum=exponential_average_factor, eps=self.eps)  # set training to True so it calculates the norm of the batch


class MyBatchNorm1d(MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class EmptyModule(nn.Module):
    def forward(self, X):
        return X


class TempSepConv_CAFF(nn.Module):
    def __init__(self, config, no_ts_features=None, no_daig_features=None, no_flat_features=None):

        super(TempSepConv_CAFF, self).__init__()
        self.task = config['task']
        self.n_layers = config['n_layers']
        self.diagnosis_size = config['diagnosis_size']
        self.main_dropout_rate = config['main_dropout_rate']
        self.temp_dropout_rate = config['temp_dropout_rate']
        self.kernel_size = config['kernel_size']
        self.temp_kernels = config['temp_kernels']
        self.last_linear_size = config['last_linear_size']
        self.no_ts_features = no_ts_features
        self.no_daig_features = no_daig_features
        self.no_flat_features = no_flat_features

        self.no_diag = config['no_diag']

        self.alpha = 100

        self.keep_prob = 1-config['main_dropout_rate'] #0.5

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=1/48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.rmsle_loss = RMSLELoss()
        self.msle_loss = MSLELoss()
        self.mse_loss = MSELoss()
        self.bce_loss = nn.BCELoss()

        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)

        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)  # removes None items from a tuple
        self.empty_module = EmptyModule()
        
        self.batchnormclass = MyBatchNorm1d
        # self.batchnormclass = nn.BatchNorm1d

        self.diagnosis_encoder = nn.Linear(in_features=self.no_daig_features, out_features=self.diagnosis_size)

        self.diagnosis_encoder1 = nn.Linear(in_features=self.no_daig_features, out_features=self.temp_kernels[0]+1)
        self.flat_encoder = nn.Linear(in_features=self.no_flat_features, out_features=self.temp_kernels[0]+1)

        
        self.bn_diagnosis_encoder = self.batchnormclass(num_features=self.diagnosis_size, momentum=0.1)  # input shape: B * diagnosis_size
        self.bn_point_last_los = self.batchnormclass(num_features=self.last_linear_size, momentum=0.1)  # input shape: (B * T) * last_linear_size
        self.bn_point_last_mort = self.batchnormclass(num_features=self.last_linear_size, momentum=0.1)  # input shape: (B * T) * last_linear_size
    
        # self.bn_diagnosis_encoder = self.empty_module
        # self.bn_point_last_los = self.empty_module
        # self.bn_point_last_mort = self.empty_module

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.final_los = nn.Linear(in_features=self.last_linear_size, out_features=1)
        self.final_mort = nn.Linear(in_features=self.last_linear_size, out_features=1)
       
        # TDSC layers settings
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1  # dilation = 1 for the first layer, after that it captures all the information gathered by previous layers
            temp_k = self.temp_kernels[i]
            
            self.layers.append({})
            if temp_k is not None:
                padding = [(self.kernel_size - 1) * dilation, 0]  # [padding_left, padding_right]
                self.layers[i]['temp_kernels'] = temp_k
                self.layers[i]['dilation'] = dilation
                self.layers[i]['padding'] = padding
                self.layers[i]['stride'] = 1

        self.layer_modules = nn.ModuleDict()

        self.Y = 0  # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        self.n = 0  # n is the layer number

        for i in range(self.n_layers):

            temp_in_channels = (self.no_ts_features + self.n) * (1 + self.Y) if i > 0 else 2 * self.no_ts_features  # (F + n) * (Y + 1)
            temp_out_channels = (self.no_ts_features + self.n) * self.layers[i]['temp_kernels']  # (F + n) * temp_kernels
            out_channels_caff = (self.no_ts_features+self.n+1)*(self.layers[i]['temp_kernels']+1)
            if self.n == 0:
                linear_input_dim = (self.no_ts_features + self.n - 1) * self.Y + 2 * self.no_ts_features + 2 + self.no_flat_features 
            else:
                linear_input_dim = (self.no_ts_features + self.n - 1) * self.Y + (self.layers[i]['temp_kernels']+1) + 2 * self.no_ts_features + 2 + self.no_flat_features  # (F + n-1) * Y + Z + 2F + 2 + no_flat_features
            linear_output_dim = (self.layers[i]['temp_kernels']+1)          

            temp = nn.Conv1d(in_channels=temp_in_channels,  # (F + n) * (Y + 1)
                             out_channels=temp_out_channels,  # (F + n) * Y
                             kernel_size=self.kernel_size,
                             stride=self.layers[i]['stride'],
                             dilation=self.layers[i]['dilation'],
                             groups=self.no_ts_features + self.n)

            caff_fc = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            
            bn_temp = self.batchnormclass(num_features=temp_out_channels, momentum=0.1)
            bn_caff = self.batchnormclass(num_features=linear_output_dim, momentum=0.1)
            
            # bn_temp = bn_point = self.empty_module  # linear module; does nothing

            A_layer = my_AFF(out_channels_caff)
            FFA_layer = my_AFF(linear_input_dim)

            self.layer_modules[str(i)] = nn.ModuleDict({
                'temp': temp,
                'bn_temp': bn_temp,
                'caff_fc': caff_fc,
                'bn_caff': bn_caff,
                'A_layer': A_layer,
                'FFA_layer': FFA_layer})

            self.Y = self.layers[i]['temp_kernels']
            self.n += 1

        # input shape: (B * T) * ((F + n) * (1 + Y) + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        # input_size = (self.no_ts_features + self.n) * (1 + self.Y) + self.diagnosis_size + self.no_flat_features
        #input_size = (self.no_ts_features + self.n) * (1 + self.Y) + self.diagnosis_size + self.no_flat_features

        input_size = (self.no_ts_features + self.n) * (1 + self.Y) + (self.n_layers * (1 + self.Y)) + self.diagnosis_size + self.no_flat_features
        if self.no_diag:
            # input_size = input_size - self.diagnosis_size
            input_size = input_size - self.diagnosis_size #input_size - self.diagnosis_size

        self.last_los_fc = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.last_mort_fc = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        return


    def tdsc_caff(self, B=None, T=None, X=None, repeat_flat=None, X_orig=None, temp=None, bn_temp=None, caff_fc=None,
                       bn_caff=None, A_layer=None, FFA_layer=None, temp_kernels=None, padding=None, prev_temp=None, prev_caff=None, m_scale_output=None,
                       caff_skip=None):

        X_padded = pad(X, padding, 'constant', 0)  # B * ((F + n) * (Y + 1)) * (T + padding)
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # B * ((F + n) * temp_kernels) * T
        
        #### Context Aware Attentive Feature Fusion (CAFF) #####
        if prev_caff is None:
            X_concat = cat(self.remove_none((prev_temp,  # (B * T) * ((F + n-1) * Y)
                                            prev_caff,  # (B * T) * 1
                                            X_orig,  # (B * T) * (2F + 2)
                                            repeat_flat)),  # (B * T) * no_flat_features
                        dim=1)  # (B * T) * (((F + n-1) * Y) + 1 + 2F + 2 + no_flat_features)
        else:
            X_concat = cat(self.remove_none((prev_temp.view(B*T,-1),  # (B * T) * ((F + n-1) * Y)
                                            prev_caff.permute(0,3,1,2).view(B*T,-1),  # (B * T) * 1
                                            X_orig,  # (B * T) * (2F + 2)
                                            repeat_flat)),  # (B * T) * no_flat_features
                        dim=1)  # (B * T) * (((F + n-1) * Y) + 1 + 2F + 2 + no_flat_features)

        X_concat, wei_1 = FFA_layer(X_concat.view(B,T,-1).permute(0,2,1)) # Step 2 Attention
        X_concat = X_concat.permute(0,2,1).view(B*T,-1)
        caff_output = self.main_dropout(bn_caff(caff_fc(X_concat)))  # (B * T) * 1
        caff_output = caff_output.view(B, T, -1).unsqueeze(2).permute(0,2,3,1)

        # Accumulate multi-scale features
        m_scale_output = cat((m_scale_output,caff_output), dim=1) if m_scale_output is not None else caff_output

        caff_skip = cat((caff_skip, prev_caff[:,:,-1,:].unsqueeze(2)), dim=1) if prev_caff is not None else caff_skip

        temp_skip = cat((caff_skip,  # B * (F + n) * 1 * T
                         X_temp.view(B, caff_skip.shape[1], temp_kernels, T)),  # B * (F + n) * temp_kernels * T
                        dim=2)  # B * (F + n) * (1 + temp_kernels) * T
        
        X_combined = self.relu(cat((temp_skip, caff_output), dim=1))  # B * (F + n) * (1 + temp_kernels) * T
        next_X = X_combined.view(B, (caff_skip.shape[1] + 1) * (1 + temp_kernels), T)  # B * ((F + n + 1) * (1 + temp_kernels)) * T
        
        next_X, wei_2 = A_layer(next_X.view(B,-1,T)) # step 4 attention
        next_X =  next_X.view(B, (caff_skip.shape[1] + 1) * (1 + temp_kernels), T)

        temp_output = X_temp.permute(0, 2, 1).contiguous().view(B * T, caff_skip.shape[1] * temp_kernels)  # (B * T) * ((F + n) * temp_kernels)

        return (temp_output,  # (B * T) * ((F + n) * temp_kernels)
                caff_output,  # (B * T) * 1
                next_X,  # B * ((F + n) * (1 + temp_kernels)) * T
                caff_skip, # caff features of the prevous layer
                m_scale_output, #  keeping track of the caff multi scale features from all layers; B * (F + n) * T
                wei_1, wei_2)  #  PWatt Attention weights 


    def forward(self, X, diagnoses, flat, time_before_pred=5):

        # flat is B * no_flat_features
        # diagnoses is B * no_daig_features
        # X is B * no_daig_features * T

        # split into features and indicator variables
        X_separated = torch.split(X[:, 1:-1, :], self.no_ts_features, dim=1)  # tuple ((B * F * T), (B * F * T))

        # prepare repeat arguments and initialise layer loop
        B, _, T = X_separated[0].shape

        repeat_flat = flat.repeat_interleave(T, dim=0)  # (B * T) * no_flat_features
        X_orig = X.permute(0, 2, 1).contiguous().view(B * T, 2 * self.no_ts_features + 2)  # (B * T) * (2F + 2)
        repeat_args = {'repeat_flat': repeat_flat,
                        'X_orig': X_orig,
                        'B': B,
                        'T': T}

        next_X = torch.stack(X_separated, dim=2).reshape(B, 2 * self.no_ts_features, T)
        caff_skip = X_separated[0].unsqueeze(2)  # ts features without indicators, keeps track of caff skip connections generated from caff module;
        temp_output = None
        caff_output = None
        m_scale_output = None
        wei_step2 = []
        wei_step4 = []
        for i in range(self.n_layers):
            kwargs = dict(self.layer_modules[str(i)], **repeat_args)
            
            temp_output, caff_output, next_X, caff_skip, m_scale_output, wei_1, wei_2 = self.tdsc_caff(X=next_X, caff_skip=caff_skip,
                                                                prev_temp=temp_output, prev_caff=caff_output,
                                                                temp_kernels=self.layers[i]['temp_kernels'],
                                                                padding=self.layers[i]['padding'],
                                                                m_scale_output= m_scale_output,
                                                                **kwargs)

            wei_step2.append(wei_1.detach().cpu())
            wei_step4.append(wei_2.detach().cpu())
            

        m_scale_output = m_scale_output.view(B,-1,T)
        if self.no_diag:
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     next_X[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1),
                                     m_scale_output[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)  # (B * (T - time_before_pred)) * (((F + n) * (1 + Y)) + no_flat_features) for tpc
        else:
            diagnoses_enc = self.relu(self.main_dropout(self.bn_diagnosis_encoder(self.diagnosis_encoder(diagnoses))))  # B * diagnosis_size
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     diagnoses_enc.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * diagnosis_size
                                     next_X[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1),
                                     m_scale_output[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)  # (B * (T - time_before_pred)) * (((F + n) * (1 + Y)) + diagnosis_size + no_flat_features) for tpc

        last_los = self.relu(self.main_dropout(self.bn_point_last_los(self.last_los_fc(combined_features))))
        last_mort = self.relu(self.main_dropout(self.bn_point_last_mort(self.last_mort_fc(combined_features))))


        los_predictions = self.hardtanh(exp(self.final_los(last_los).view(B, T - time_before_pred)))  # B * (T - time_before_pred)
        mort_predictions = self.sigmoid(self.final_mort(last_mort).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions, wei_step2, wei_step4

    def loss(self, y_hat_los, y_hat_mort, y_los, y_mort, mask, seq_lengths, device, sum_losses, loss_type):
        # mortality loss
        if self.task == 'mortality':
            loss = self.bce_loss(y_hat_mort, y_mort) * self.alpha
        # LoS loss
        else:
            bool_type = torch.cuda.BoolTensor if device == torch.device('cuda:3') else torch.BoolTensor
            if loss_type == 'rmsle':
                los_loss = self.rmsle_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            if loss_type == 'msle':
                los_loss = self.msle_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            elif loss_type == 'mse':
                los_loss = self.mse_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)

            loss = los_loss
        return loss
