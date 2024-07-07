import numpy as np
import os
import math
import warnings
import torch
import torch.nn.functional as F

def read_radar_bins(path_bins):
    # read radar bin files
    range_bins = np.loadtxt(os.path.join(path_bins, "rangebins.txt"), dtype=float, delimiter=None, 
                            converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
    azi_bins = np.loadtxt(os.path.join(path_bins, "azimuthbins.txt"), dtype=float, delimiter=None, 
                          converters=None, skiprows=0, usecols=None, unpack=False,
                          ndmin=0, encoding='bytes', max_rows=None)
    velocity_bins = np.loadtxt(os.path.join(path_bins, "velocitybins.txt"), dtype=float, delimiter=None, 
                               converters=None, skiprows=0, usecols=None, unpack=False,
                               ndmin=0, encoding='bytes', max_rows=None)

    return range_bins, azi_bins, velocity_bins

def radar_bins_interpolate(bins, interp_factor):
    interpolateBins = F.interpolate(torch.tensor(bins).unsqueeze(1).unsqueeze(0).unsqueeze(0), (int(bins.shape[0] * interp_factor), 1), mode='bilinear', align_corners=True).squeeze(0).squeeze(0).squeeze(1).numpy()
    return interpolateBins

def read_radar_cube(path, dType, data_shape):
    # read radar data from the bin file
    radar_cube = np.fromfile(path, dtype=dType)
    radar_cube = np.reshape(radar_cube, [data_shape[0], data_shape[2], data_shape[1]], order='C')
    radar_cube = np.transpose(radar_cube, (0, 2, 1)) # [doppler, range, azi]
    
    return radar_cube

def compute_ego_velocity_bin(ego_velocity, velocity_bins):
    # naively, the bin of the ego velocity can be found using argmin(abs(egovelocity - velocitybins))
    # however, due to velocity ambiguity, velocities larger than max(velocityBins) wrap around the range
    # we will estimate if there is a wrap around, what is the appropriate bin
    if ego_velocity <= velocity_bins[-1]:
        bin = np.argmin(np.abs(ego_velocity - velocity_bins))
        ambiguous_ego_velocity = ego_velocity
    else:
        wraps = (ego_velocity - velocity_bins[-1]) / (velocity_bins[-1] - velocity_bins[0])
        ambiguous_ego_velocity = velocity_bins[0] + math.modf(wraps)[0]*(velocity_bins[-1] - velocity_bins[0])
        bin = np.argmin(np.abs(ambiguous_ego_velocity - velocity_bins))

    return (bin, ambiguous_ego_velocity)

def ego_correction(radar_cube, maxEnergyIdx):
    # find the velocity bin with most energy, re-center the data around this slice and remove unwanted velocities
    radar_cube = np.roll(radar_cube, int(radar_cube.shape[0] / 2 - maxEnergyIdx), axis=0)

    return radar_cube

def gt_occupancy_map(labels, range_bins, azi_bins, max_range=40):
    # creates an ground truth occupancy map
    occ_map = np.zeros((range_bins.shape[0], azi_bins.shape[0]))

    for i in range(0,labels.shape[1]):
        x = labels[1, i]
        y = labels[2, i]

        rho = np.sqrt(x**2+y**2)
        azi = -np.arctan2(y,x) + np.pi/2
        if rho <= max_range:
            cIdx = np.argmin(abs(azi_bins - azi), axis=None, out=None)
            rIdx = np.argmin(abs(range_bins - rho), axis=None, out=None)
            occ_map[rIdx,cIdx] = 1

    occ_map = np.expand_dims(occ_map, axis=0)

    return occ_map

def load_test_sequences_list(path_test_list):
    testList = []
    f = open(path_test_list, 'r')
    for line in f.readlines():
        line = line.split(' ')
        line = line[1]
        testList.append(line.replace('\n', ''))
    f.close()

    return testList

def normalize(v):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.asarray(v, dtype=np.float)
        minVal = np.min(v)
        if minVal == 0:
            minVal = np.finfo(float).eps
        v -= minVal
        v = np.divide( v , np.max(v) )
    return v