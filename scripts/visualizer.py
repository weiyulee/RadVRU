import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from torch.utils.data import DataLoader, SequentialSampler
import tqdm
import pickle

from dataset import RadVRU_dataset
import util

def main():

    path_dataset = '/scratch/welee/dataset/Fusion/RadVRU/' 

    path_test_seqs = os.path.join(path_dataset, "test_sequences.txt")
    test_list = util.load_test_sequences_list(path_test_seqs)

    path_imdb_pkl = os.path.join(path_dataset, "RadVRU_db.pkl")
    if os.path.exists(path_imdb_pkl):
        with open(path_imdb_pkl, 'rb') as f:  
            imdb_labels, imdb_vids, imdb_ego_velocity = pickle.load(f)
    else:
        print("Cannot find {}.".format(path_imdb_pkl))
        return

    # only visualize the sequences in the test set
    val_labels = []
    val_vids = []
    val_ego_velocity = []
    for idx, v in enumerate(imdb_vids):
        sequenceName = os.path.join(v[0], 'rgb', v[1])
        if sequenceName in test_list:
            val_labels.append(imdb_labels[idx])
            val_vids.append(imdb_vids[idx])
            val_ego_velocity.append(imdb_ego_velocity[idx])

    val_dataset = RadVRU_dataset(dataset_dir=path_dataset,
                                 imdb_labels=val_labels,
                                 vids=val_vids,
                                 ego_velocity=val_ego_velocity,
                                 max_range=40)

    n_val = int(len(val_labels))

    val_sampler  = SequentialSampler(np.ones(int(n_val)))
    val_loader   = DataLoader(val_dataset, sampler=val_sampler, batch_size=1, num_workers=4, pin_memory=True)

    # prepare axes for plot
    aziBinsPlot = np.concatenate((val_dataset.azi_bins, [np.pi/2]), axis=0)
    aziBinsPlot = (aziBinsPlot[:-1] + aziBinsPlot[1:]) / 2        
    aziBinsPlot = np.rad2deg(aziBinsPlot)
    x_azi_ticklabels = [aziBinsPlot[i] for i in range(0, 128, 16)]      

    velocity_bins_plot = val_dataset.velocity_bins
    x_v_ticklabels = [velocity_bins_plot[i] for i in range(0, 128, 16)]      

    valid_range = val_dataset.range_bins <= 40
    valid_range_bins = len(np.nonzero(valid_range)[0])        
    f = interpolate.interp1d(np.arange(0, valid_range_bins), val_dataset.range_bins[:valid_range_bins][::-1], fill_value='extrapolate')
    range_bins_plot = f(np.arange(0, valid_range_bins, valid_range_bins/128))
    y_ticklabels = [range_bins_plot[i] for i in range(0, 128, 16)]

    # set plot details
    plt.close('all')
    fig, ax = plt.subplots(2,2, figsize=(15, 10))
    init_img = np.zeros(shape=(128, 128), dtype=np.uint8)
    ax00 = ax[0,0].imshow(init_img, vmin=0, vmax= 1, cmap='jet')        
    ax[0,0].set_title('Range-Azimuth Singals', fontsize=22)
    ax[0,0].set_ylabel('Range [m]', fontsize=12)
    ax[0,0].set_yticks(np.arange(0, 128, 16))
    ax[0,0].set_yticklabels(np.array(np.around(y_ticklabels), dtype=int), fontsize=16)
    ax[0,0].set_xlabel('Azimuth [degrees]', fontsize=12)
    ax[0,0].set_xticks(np.arange(1, 129, 16))
    ax[0,0].set_xticklabels(np.array(x_azi_ticklabels, dtype=int), fontsize=16)       
    cb00 = fig.colorbar(ax00, ax=ax[0,0], location='left')  
    cb00.ax.tick_params(labelsize=12)
    cb00.set_label('Strength of normailized radar signals', size=12)

    ax01 = ax[0,1].imshow(init_img, vmin=0, vmax= 1, cmap='jet')        
    ax[0,1].set_title('Range-Doppler Singals', fontsize=22)
    ax[0,1].set_ylabel('Range [m]', fontsize=12)
    ax[0,1].set_yticks(np.arange(0, 128, 16))
    ax[0,1].set_yticklabels(np.array(np.around(y_ticklabels), dtype=int), fontsize=16)
    ax[0,1].set_xlabel('Doppler [m/s]', fontsize=12)
    ax[0,1].set_xticks(np.arange(1, 129, 16))
    ax[0,1].set_xticklabels(np.array(x_v_ticklabels, dtype=int), fontsize=16)       
    cb01 = fig.colorbar(ax01, ax=ax[0,1], location='left')  
    cb01.ax.tick_params(labelsize=12)
    cb01.set_label('Strength of normailized radar signals', size=12)

    ax10 = ax[1,0].imshow(init_img, vmin=0, vmax= 1, cmap='jet')        
    ax[1,0].set_title('Vision Feedforward', fontsize=22)
    ax[1,0].set_ylabel('Range [m]', fontsize=12)
    ax[1,0].set_yticks(np.arange(0, 128, 16))
    ax[1,0].set_yticklabels(np.array(np.around(y_ticklabels), dtype=int), fontsize=16)
    ax[1,0].set_xlabel('Azimuth [degrees]', fontsize=12)
    ax[1,0].set_xticks(np.arange(1, 129, 16))
    ax[1,0].set_xticklabels(np.array(x_azi_ticklabels, dtype=int), fontsize=16)                            
    cb10 = fig.colorbar(ax10, ax=ax[1,0], location='left')                     
    cb10.ax.tick_params(labelsize=12)
    cb10.set_label('Confidence of presence', size=12)
    
    ax11 = ax[1,1].imshow(init_img, vmin=0, vmax= 1, cmap='jet')        
    ax[1,1].set_title('Ground truth', fontsize=22)
    ax[1,1].set_ylabel('Range [m]', fontsize=12)
    ax[1,1].set_yticks(np.arange(0, 128, 16))
    ax[1,1].set_yticklabels(np.array(np.around(y_ticklabels), dtype=int), fontsize=16)
    ax[1,1].set_xlabel('Azimuth [degrees]', fontsize=12)
    ax[1,1].set_xticks(np.arange(1, 129, 16))
    ax[1,1].set_xticklabels(np.array(x_azi_ticklabels, dtype=int), fontsize=16)             
    fig.tight_layout()

    with tqdm.tqdm(total=n_val, desc='Visualizing') as pbar:
        for batch in val_loader:
            radar_cube = batch['radar_cube']  
            rgb_image = batch['rgb_image']  
            vf = batch['vf']  
            gt_occ_map = batch['gt_occ_map'] 
            raw_labels = batch['raw_labels']

            # Visualize rgb image
            cv2.namedWindow('RGB image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RGB image', 960, 540)
            cv2.imshow('RGB image', rgb_image.squeeze().numpy().astype(np.uint8))                    
            cv2.waitKey(1)

            # Visualize RA map
            ramap = radar_cube[0, :, :valid_range_bins, :].mean(axis=0)
            ax00.set_data(cv2.resize(np.flipud(util.normalize(ramap)), [128, 128]))

            # Visualize RD map
            rdmap = radar_cube[0, :, :valid_range_bins, :].mean(axis=-1)
            rdmap = np.transpose(rdmap, (1, 0))
            ax01.set_data(cv2.resize(np.fliplr(np.flipud(util.normalize(rdmap))), [128, 128]))

            # Visualize vision feedforward
            ax10.set_data(cv2.resize(np.flipud(vf.squeeze()), [128, 128]))

            # Draw Ground truths
            ax11.set_data(cv2.resize(np.flipud(gt_occ_map.squeeze()), [128, 128]))  
                        
            plt.pause(0.001)

            pbar.update(1)
    return

if __name__ == '__main__':
    main()