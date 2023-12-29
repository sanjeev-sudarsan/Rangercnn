import numpy as np
import torch
import matplotlib.pyplot as plt

HEIGHT = np.array(
  [0.20966667, 0.2092, 0.2078, 0.2078, 0.2078,
   0.20733333, 0.20593333, 0.20546667, 0.20593333, 0.20546667,
   0.20453333, 0.205, 0.2036, 0.20406667, 0.2036,
   0.20313333, 0.20266667, 0.20266667, 0.20173333, 0.2008,
   0.2008, 0.2008, 0.20033333, 0.1994, 0.20033333,
   0.19986667, 0.1994, 0.1994, 0.19893333, 0.19846667,
   0.19846667, 0.19846667, 0.12566667, 0.1252, 0.1252,
   0.12473333, 0.12473333, 0.1238, 0.12333333, 0.1238,
   0.12286667, 0.1224, 0.12286667, 0.12146667, 0.12146667,
   0.121, 0.12053333, 0.12053333, 0.12053333, 0.12006667,
   0.12006667, 0.1196, 0.11913333, 0.11866667, 0.1182,
   0.1182, 0.1182, 0.11773333, 0.11726667, 0.11726667,
   0.1168, 0.11633333, 0.11633333, 0.1154])
ZENITH = np.array([
  0.03373091, 0.02740409, 0.02276443, 0.01517224, 0.01004049,
  0.00308099, -0.00155868, -0.00788549, -0.01407172, -0.02103122,
  -0.02609267, -0.032068, -0.03853542, -0.04451074, -0.05020488,
  -0.0565317, -0.06180405, -0.06876355, -0.07361411, -0.08008152,
  -0.08577566, -0.09168069, -0.09793721, -0.10398284, -0.11052055,
  -0.11656618, -0.12219002, -0.12725147, -0.13407038, -0.14067839,
  -0.14510716, -0.15213696, -0.1575499, -0.16711043, -0.17568678,
  -0.18278688, -0.19129293, -0.20247031, -0.21146846, -0.21934183,
  -0.22763699, -0.23536977, -0.24528179, -0.25477201, -0.26510582,
  -0.27326038, -0.28232882, -0.28893683, -0.30004392, -0.30953414,
  -0.31993824, -0.32816311, -0.33723155, -0.34447224, -0.352908,
  -0.36282001, -0.37216965, -0.38292524, -0.39164219, -0.39895318,
  -0.40703745, -0.41835542, -0.42777535, -0.43621111
])
INCL = -ZENITH


class range_image_creator():
  def __init__(self, H=64, W=2048, cW =512, remission= False,cfg=None):
    """
    Parameters:
        H (int) = Height of the range image.
        W (int) = Width of the range image.
        cW (int) = Cropped width of the range image.
        remission (boolean) = If remission is required.
    """
    
    self.H = H
    self.W = W
    self.cW = cW
    self.cfg = cfg
    self.remission = remission
    if self.remission:
      self.intensity = np.zeros((H, W))
    self.spherical_rangemap = np.zeros((H, W))
    self.polar_rangemap = np.zeros((H, W, 2))
    self.cartesian_rangemap = np.zeros((H, W, 3))

  def crop_range_image(self, range_image):
    """
    Crops the range image
    Parameters:
        range_image (numpy.ndarray) = The range image to be cropped.

    Returns:
        new_range_image (numpy.ndarray): The cropped image.
    """
    
    mid = self.W // 2
    crop = self.cW // 2
    beg = mid - crop
    end = mid + crop
    try:
      new_range_image = range_image[:48, beg:end]
    except:
      new_range_image = range_image[:48, beg:end, :]
    return new_range_image



  def create(self, pc):
    """
    Creates range images from a point cloud.
    Parameters:
        pc (numpy.ndarray) = Point cloud.

    """
    
    if self.remission:
      intensity = pc[:,3]
    else:
      intensity = None
    pc = pc[:,:3]

    # Determine the indices of all the points in the range image.
    xy_norm = np.linalg.norm(pc[:, :2], ord=2, axis=1)
    error_list = []
    for i in range(len(INCL)):
      h = HEIGHT[i]
      theta = INCL[i]
      error = np.abs(theta - np.arctan2(h - pc[:, 2], xy_norm))
      error_list.append(error)
    all_error = np.stack(error_list, axis=-1)
    row_inds = np.argmin(all_error, axis=-1)
    azi = np.arctan2(pc[:, 1], pc[:, 0])
    width = self.W
    col_inds = width - 1.0 + 0.5 - (azi + np.pi) / (2.0 * np.pi) * width
    col_inds = np.round(col_inds).astype(np.int32)
    col_inds[col_inds == width] = width - 1
    col_inds[col_inds < 0] = 0

    point_range = np.linalg.norm(pc[:, :3], axis=1, ord=2)
    order = np.argsort(-point_range)
    point_range = point_range[order]
    pc = pc[order]
    row_inds = row_inds[order]
    col_inds = col_inds[order]
    if self.remission:
      intensity = intensity[order]
      self.intensity[row_inds, col_inds] = intensity
    self.spherical_rangemap[row_inds, col_inds] = point_range
    self.cartesian_rangemap[row_inds, col_inds] = pc[:, :3]
    self.col_inds = col_inds
    self.row_inds = row_inds

    if self.W != self.cW:
      self.spherical_rangemap = self.crop_range_image(self.spherical_rangemap)
      if self.remission:
        self.intensity = self.crop_range_image(self.intensity)
      self.cartesian_rangemap = self.crop_range_image(self.cartesian_rangemap)


#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
import sys
import os

from numpy.testing._private.utils import decorate_methods
import torch
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
  sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask


  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, p):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()
    '''
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    '''
    self.scan = p
    # scan = scan.reshape((-1, 4))

    # put in attribute
    points = self.scan[:, 0:3]    # get xyz
    #remissions = self.scan[:, 3]  # get remission
    remissions = None
    out_dict = self.set_points(points, remissions)
    return out_dict

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      out_dict = self.do_range_projection()

    return out_dict

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1) + 1e-7
    polar_depth = np.linalg.norm(self.points[:,:2], 2, axis=1)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    proj_y *= self.proj_H  # in [0.0, H]
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # get projections in image coords

    rad = 1.39626
    proj_x = 0.5 * ((2 * yaw) / rad + 1.0)  # in [0.0, 1.0]


      # scale to image size using angular resolution
    proj_x *= self.proj_W  # in [0.0, W]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)
    #self.unproj_polar_range = np.copy(polar_depth)

    # order in decreasing depth，
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y_z = proj_y[order]
    proj_x_z = proj_x[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    scan = self.scan[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)
    out_dict = {
      'range': self.proj_range,
      'ori_xyz': self.proj_xyz,
      'ori_r': self.proj_remission,
      'idx': self.proj_idx,
      'mask': self.proj_mask,
      'p_y': proj_y,
      'p_x': proj_x
    }

    return out_dict

