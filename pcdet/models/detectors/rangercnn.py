from .detector3d_template import Detector3DTemplate
import torch

from ...ops.roiaware_pool3d import roiaware_pool3d_utils



class RangeRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):

        #batch_dict['range_scan'] = batch_dict['range_scan'].permute(0,2,1).contiguous() # B*C*N   1*4*N
        batch_dict['rangemap_xyz'] = batch_dict['rangemap_xyz'].permute(0,3,1,2).contiguous()
        batch_dict['rangemap_spherical'] = batch_dict['rangemap_spherical'].unsqueeze(1).contiguous()

        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        if len(self.dataset.dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list) >= 4:
            batch_dict['intensity'] = batch_dict['intensity'].unsqueeze(1).contiguous()
            batch_dict.update({'range_in': torch.cat((batch_dict['rangemap_spherical'],batch_dict['rangemap_xyz'], batch_dict['intensity']),1)})
            del batch_dict['intensity']
        else:
            batch_dict.update({'range_in': torch.cat(
                (batch_dict['rangemap_spherical'], batch_dict['rangemap_xyz']), 1)})

        del batch_dict['rangemap_spherical']


        for cur_module in self.module_list:

            
            batch_dict = cur_module(batch_dict)


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        #loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn #+ loss_rcnn
        return loss, tb_dict, disp_dict
