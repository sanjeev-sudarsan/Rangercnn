"""
to generate bird eye view, we have to filter point cloud
first. which means we have to limit coordinates


"""
import numpy as np

import os
import torch
import sys
import torch.nn as nn

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
  sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import matplotlib.pyplot as plt


# POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1] res: 0.1  800*704  stride: 4 200*176

#[0,69.12] [-39.68,39.68]  bev : 496*432 y*x

# image size would be 800*704


class RANGE2BEV(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg  # range, res
        self.x_range = self.model_cfg['X_RANGE']
        self.y_range = self.model_cfg['Y_RANGE']
        self.z_range = self.model_cfg['Z_RANGE']
        self.resolution = self.model_cfg['RESOLUTION']
        self.z_resolution = 0.8
        d = int((self.z_range[1] - self.z_range[0]) / self.z_resolution)



        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES


        #self.avg_pool = nn.AvgPool2d((1,d))

    def forward(self,batch_dict):

        bc = batch_dict['batch_size']
        range_up = batch_dict['range_res']
        _,c,rh,rw = range_up.shape

        w = int((self.y_range[1] - self.y_range[0])/self.resolution) # 800
        h = int((self.x_range[1] - self.x_range[0])/self.resolution) # 704
        d = int((self.z_range[1] - self.z_range[0])/self.z_resolution) # 704

        bev = torch.zeros([bc , w , h , c , d])
        bev = bev.cuda()


        for b in range(bc):
            scan = batch_dict['rangemap_xyz'][b, :, :, :]
            scan = scan.reshape(3, rh * rw).permute(1, 0).contiguous()
            range_f = range_up[b, :, :, :]
            point = range_f.reshape(64, rh * rw).permute(1, 0).contiguous()

            bev_list = []

            x_points = scan[:, 0]
            y_points = scan[:, 1]
            z_points = scan[:, 2]

            for i, height in enumerate(torch.arange(self.z_range[0], self.z_range[1], self.z_resolution)):
                z_filt = torch.logical_and((z_points >= height),
                                           (z_points < height + self.z_resolution))
                im = torch.zeros((w, h, c), dtype=torch.float32)
                im = im.cuda()
                if torch.all(z_filt==False):
                    im = im.unsqueeze(3)
                    bev_list.append(im)
                    continue

                indices = torch.nonzero(z_filt)
                indices = torch.flatten(indices)

                # KEEPERS
                xi_points = x_points[indices]
                yi_points = y_points[indices]
                zi_points = z_points[indices]
                i_point = point[indices]



                x_img = (-yi_points / self.resolution).long()  # x axis is -y in LIDAR
                y_img = (-xi_points / self.resolution).long()  # y axis is -x in LIDAR

                x_img -= int(np.floor(self.y_range[0] / self.resolution))
                y_img += int(np.floor(self.x_range[1] / self.resolution))

                x_max = int((self.y_range[1] - self.y_range[0]) / self.resolution - 1)  # 799
                y_max = int((self.x_range[1] - self.x_range[0]) / self.resolution - 1)  # 703

                x_img = torch.clamp(x_img, 0, x_max)
                y_img = torch.clamp(y_img, 0, y_max)

                if i_point.is_cuda:
                    im = im.cuda()

                im[x_img, y_img] = i_point

                im = torch.flip(im, dims=[0, 1])


                im = im.unsqueeze(3)

                bev_list.append(im)


            bev_tens = torch.cat(bev_list,dim=3)



            bev[b] = bev_tens



        bev = bev.permute(0,3,4,1,2).contiguous()

        batch_dict['voxel_features'] =  bev

        return batch_dict
