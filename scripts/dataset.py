import util
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class RadVRU_dataset(Dataset):
    """Radar dataset."""

    def __init__(self, dataset_dir, imdb_labels, vids, ego_velocity, max_range=40):
        """
        Args:
            autoLabelsPath (string): Path to the annotations.
            root_dir (string): Directory with all the images.
        """

        # init parameters
        self.range_resize_factor = 1
        self.azi_resize_factor = 8
        self.velocityResizeFactor = 1
        self.max_range = max_range

        # read bin information
        path_bins = os.path.join(dataset_dir, vids[0][0], 'radar', vids[0][1]) # get the path from the first entry
        range_bins_ori, azi_bins_ori, velocity_bins_ori = util.read_radar_bins(path_bins)       
        
        # original radar cube resolution is [doppler, range, azi] = [128, 128, 16]
        # resize the radar cube to [128, 128, 128] for better visualization
        range_bins = util.radar_bins_interpolate(range_bins_ori, self.range_resize_factor)
        azi_bins = util.radar_bins_interpolate(azi_bins_ori, self.azi_resize_factor)
        velocity_bins = util.radar_bins_interpolate(velocity_bins_ori, self.velocityResizeFactor)

        self.imdb_labels = imdb_labels
        self.vids = vids
        self.ego_velocity = ego_velocity

        self.dataset_dir = dataset_dir
        
        self.range_bins_ori = range_bins_ori
        self.azi_bins_ori = azi_bins_ori
        self.velocity_bins_ori = velocity_bins_ori
        
        self.range_bins = range_bins
        self.azi_bins = azi_bins
        self.velocity_bins = velocity_bins

    def __len__(self):
        return len(self.imdb_labels)

    def __getitem__(self, idx):

        labels = self.imdb_labels[idx]

        radar_data_shape = [self.velocity_bins_ori.shape[0], self.range_bins_ori.shape[0], self.azi_bins_ori.shape[0]] # 128x128x16
        vf_data_shape = [self.range_bins_ori.shape[0], self.azi_bins.shape[0]] # 1x128x128        

        # get radar, vision feedforward, and rgb paths
        path_ramap = os.path.join(self.dataset_dir, self.vids[idx][0], 'radar', self.vids[idx][1], self.vids[idx][2])
        path_vf = os.path.join(self.dataset_dir, self.vids[idx][0], 'vf', self.vids[idx][1], "yoloOCCmap_" + self.vids[idx][2][5:] + '.bin')
        path_rgb = os.path.join(self.dataset_dir, self.vids[idx][0], 'rgb', self.vids[idx][1], self.vids[idx][2][5:] + '.jpg')

        # read the rgb data
        rgb_image = np.array(Image.open(path_rgb))
        rgb_image = rgb_image[:, :, ::-1].copy()

        # read the vf data
        vf = np.reshape(np.fromfile(path_vf), vf_data_shape)

        # read the radar cube 
        radar_cube = util.read_radar_cube(path_ramap, np.float32, radar_data_shape)

        # get lidar ego-velocity
        ego_velocity = self.ego_velocity[idx] # sample ego-velocity computed from lidar in m/s
        (ego_velocity_bin, ambiguous_ego_velocity) = util.compute_ego_velocity_bin(ego_velocity, self.velocity_bins)
        # # we're interested in the doppler bin of the static world, so:
        static_bin = self.velocity_bins.shape[0] - ego_velocity_bin
        radar_cube = util.ego_correction(radar_cube, static_bin)

        radar_cube[:, 0:3, :]  = np.zeros((radar_cube.shape[0], 3, radar_cube.shape[2]))
        radar_cube[:, -3:, :] = np.zeros((radar_cube.shape[0], 3, radar_cube.shape[2]))
            
        # for ground truth
        gt_occ_map = util.gt_occupancy_map(labels, self.range_bins, self.azi_bins, max_range=self.max_range)
        # we have to send a fixed size raw_labels in order for it to collate well, not used entries are marked with -1 and will be removed when needed
        labels = np.expand_dims(labels, axis=0)
        raw_labels = np.zeros([labels.shape[0], labels.shape[1], 100]) - 1 
        raw_labels[:, :, 0:labels.shape[2]] = labels

        data = {'idx': idx, 'rgb_image': rgb_image, 'radar_cube': radar_cube, 'vf': vf, 'gt_occ_map': gt_occ_map, 'raw_labels': raw_labels}

        return data