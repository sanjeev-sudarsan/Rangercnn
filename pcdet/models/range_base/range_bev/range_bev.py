import torch
import torch.nn as nn



class RangeBEV(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg

        input_c = self.model_cfg['INPUT_C']

        if self.model_cfg.get('LAYER_NUM', None) is not None:
            assert len(self.model_cfg.LAYER_NUM) == len(self.model_cfg.NUM_FILTERS) == len(
                self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_PADDING)
            # print('True')
            layer_nums = self.model_cfg.LAYER_NUM
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
            num_pad = self.model_cfg.NUM_PADDING
            c_in_list = [input_c, *num_filters]
        else:
            layer_nums = layer_strides = num_filters = []

        num_levels = len(layer_nums)

        self.blocks = nn.ModuleList()
        bev_out = []
        down_factor = 2
        for idx in range(num_levels):
            cin = c_in_list[idx]
            cout = c_in_list[idx + 1]
            bev_out.append(cout)
            # kernel = down_layer_strides[idx]
            if idx == 0:
                cur_layers = [
                    nn.Conv2d(cin, cout, kernel_size=layer_strides[idx], stride=(1, 1), bias=False,
                              padding=num_pad[idx]),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                ]
            else:
                cur_layers = [
                    nn.Conv2d(cin, cout, kernel_size=layer_strides[idx], stride=(2, 2), bias=False, padding=num_pad[idx]),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                ]
            self.blocks.append(nn.Sequential(*cur_layers))

        self.num_bev_features = sum(bev_out)

    def forward(self, data_dict):

        x = data_dict['spatial_features']

        out = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            out.append(x)

        up_temp_cat_f_2 = nn.functional.interpolate(out[1], scale_factor=(2, 2), mode='bilinear')
        up_temp_cat_f_3 = nn.functional.interpolate(out[2], scale_factor=(4, 4), mode='bilinear')
        data_dict['spatial_features'] = torch.cat((out[0], up_temp_cat_f_2,up_temp_cat_f_3), dim=1)

        return data_dict