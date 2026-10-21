# Portions of this file are derived from DeepSense6G_TII:
# https://github.com/DeepSenseChallengeTeam/DeepSense6G_TII
# Copyright (c) 2022 Qiyang Zhao, Yu Tian, Zine el abidine Kherroubi,
# and Fouzi Boukhalfa. Licensed under the MIT License. See
# THIRD_PARTY_LICENSES/DeepSense6G_TII_LICENSE.md.

import os
from PIL import Image
import pandas as pd

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import open3d as o3d
from scipy import stats
from sklearn.preprocessing import normalize
import utm
import re

class CARLA_Data(Dataset):
    def __init__(self, root, root_csv, config, test=False):

        self.root_csv = root_csv
        self.dataframe = pd.read_csv(root+root_csv)
        self.root=root
        self.seq_len = config.seq_len
        self.gps_data = []
        self.pos_input_normalized = Normalize_loc(
            root,
            self.dataframe,
            angle_norm=config.angle_norm,
            seq_len=self.seq_len,
            root_csv=self.root_csv,
        )
        self.test = test
        self.add_velocity = config.add_velocity
        self.custom_FoV_lidar = config.custom_FoV_lidar
        self.degradation_map = {}
        if bool(getattr(config, 'use_degradation', 0)):
            self.degradation_map.update(self._parse_degradation_map(
                getattr(config, 'degradation', ''),
                getattr(config, 'degraded_modalities', 'image'),
            ))
        if self.test:
            self.degradation_map.update(self._parse_degradation_map(
                getattr(config, 'test_degradation', ''),
                getattr(config, 'test_degraded_modalities', 'image'),
            ))
        self.degradation_strict = bool(getattr(config, 'degradation_strict', 1))
        self.test_degradation_strict = bool(getattr(config, 'test_degradation_strict', 1))

    def _parse_degradation_map(self, spec, default_modalities):
        spec = str(spec or '').strip()
        if not spec:
            return {}
        alias_map = {
            'image': 'image',
            'img': 'image',
            'camera': 'image',
            'cam': 'image',
            'lidar': 'lidar',
            'lid': 'lidar',
            'radar': 'radar',
            'rad': 'radar',
        }
        degradation_map = {}
        entries = [item.strip() for item in re.split(r'[;,]', spec) if item.strip()]
        for entry in entries:
            if ':' in entry or '=' in entry:
                sep = ':' if ':' in entry else '='
                modality, degradation = [part.strip().lower() for part in entry.split(sep, 1)]
                if modality not in alias_map:
                    raise ValueError(f"Unsupported degraded modality '{modality}'.")
                degradation_map[alias_map[modality]] = degradation
            else:
                modalities = [
                    item.strip().lower()
                    for item in str(default_modalities or 'image').replace('+', ',').split(',')
                    if item.strip()
                ]
                for modality in modalities:
                    if modality not in alias_map:
                        raise ValueError(f"Unsupported degraded modality '{modality}'.")
                    degradation_map[alias_map[modality]] = entry.lower()
        return degradation_map

    def _apply_degradation_path(self, rel_path, modality):
        degradation = self.degradation_map.get(modality, '')
        if not degradation:
            return rel_path
        base_dirs = {
            'image': 'camera_data',
            'lidar': 'lidar_data',
            # Radar CSV paths are converted to radar_data_ang before this
            # function is called, so rewrite that actual input directory.
            'radar': 'radar_data_ang',
        }
        base_dir = base_dirs[modality]
        target_dir = degradation if degradation.startswith(base_dir) else f'{base_dir}_{degradation}'
        degraded_path = re.sub(
            rf'(?<![A-Za-z0-9_]){re.escape(base_dir)}(?=[/\\])',
            target_dir,
            str(rel_path),
            count=1,
        )
        strict = self.test_degradation_strict if self.test else self.degradation_strict
        if degraded_path == rel_path and strict:
            raise ValueError(
                f"Could not apply degradation '{degradation}' to {modality} path: {rel_path}"
            )
        return degraded_path

    def _get_selected_instance_indices(self):
        """Return the last seq_len temporal slots from the fixed 5-frame layout."""
        total_instances = 5
        if self.seq_len < 1 or self.seq_len > total_instances:
            raise ValueError(
                f"Unsupported seq_len={self.seq_len}. Expected an integer in [1, {total_instances}] "
                "for the current 5-frame dataset layout."
            )
        start_idx = total_instances - self.seq_len + 1
        return [str(i) for i in range(start_idx, total_instances + 1)]

    def __len__(self):
        """Returns the length of the dataset. """
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lidars'] = []
        data['radars'] = []
        data['gps'] = self.pos_input_normalized[index, :, :].copy()
        data['scenario'] = []
        data['loss_weight'] = []

        # Resolve the mmWave power-vector path used during evaluation.
        pwr_rel = self.dataframe['unit1_pwr_60ghz'][index]  # e.g. "./scenario31/unit1/mmWave_data/mmWave_power_183.txt"
        if isinstance(pwr_rel, str) and pwr_rel.startswith("./"):
            pwr_abs = os.path.join(self.root, pwr_rel[2:])
        else:
            pwr_abs = os.path.join(self.root, pwr_rel)
        data['pwr_path'] = pwr_abs
        data['pwr_path_rel'] = pwr_rel

        PT=[]
        add_fronts = []
        add_lidars = []
        add_radars = []
        # Select the last seq_len temporal instances from the fixed 5-frame layout.
        # Indices 1..5 correspond to t-4 .. t.
        instanceidx = self._get_selected_instance_indices()
        for stri in instanceidx:
            add_fronts.append(
                self._apply_degradation_path(self.dataframe['unit1_rgb_'+stri][index], 'image')
            )
            # lidar data
            lidar_dir = self.dataframe['unit1_lidar_'+stri][index]
            add_lidars.append(self._apply_degradation_path(lidar_dir, 'lidar'))
            # radar data
            radar_dir = self.dataframe['unit1_radar_'+stri][index]
            radar_dir = re.sub('radar_data/', 'radar_data_ang/', radar_dir)
            add_radars.append(self._apply_degradation_path(radar_dir, 'radar'))

        active_seq_len = len(instanceidx)

        # check which scenario is the data sample associated 
        scenarios = ['scenario31', 'scenario32', 'scenario33', 'scenario34']
        loss_weights = [1.0, 1.0, 1.0, 1.0]

        for i in range(len(scenarios)):
            s = scenarios[i]
            if s in self.dataframe['unit1_rgb_5'][index]:
                data['scenario'] = s
                data['loss_weight'] = loss_weights[i]
                break

        for i in range(active_seq_len):
            imgs = np.array(Image.open(self.root + add_fronts[i]).resize((256, 256)))
            # radar data
            radar_ang1 = np.load(self.root + add_radars[i])
            data['fronts'].append(torch.from_numpy(np.transpose(imgs, (2, 0, 1))))
            radar_ang = np.expand_dims(radar_ang1, 0)

            if self.add_velocity:
                radar_vel_path = re.sub(
                    r'(?<![A-Za-z0-9_])radar_data_ang(?=(?:_[^/\\]+)?[/\\])',
                    'radar_data_vel',
                    add_radars[i],
                    count=1,
                )
                if radar_vel_path == add_radars[i]:
                    raise ValueError(
                        f"Could not derive radar velocity path from: {add_radars[i]}"
                    )
                radar_vel1 = np.load(self.root + radar_vel_path)
                radar_vel = np.expand_dims(radar_vel1, 0)
                radar_tensor = np.concatenate([radar_ang, radar_vel], 0)
            else:
                radar_tensor = radar_ang
            data['radars'].append(torch.from_numpy(radar_tensor.astype(np.float32)))

            # lidar data
            PT = np.asarray(o3d.io.read_point_cloud(self.root+add_lidars[i]).points)
            PT = lidar_to_histogram_features(PT, add_lidars[i],custom_FoV=self.custom_FoV_lidar)
            # lidar data
            # lidar_path = self.root + add_lidars[i]
            # lidar_path = lidar_path.replace(
            #     'lidar_data/', 'lidar_bev/'
            # )
            # lidar_path = lidar_path.replace('.ply', '.npy')
            # PT = np.load(lidar_path)

            data['lidars'].append(PT)

        if index == 0 and i == 0:
            print('Lidar feature shape:', PT.shape, PT.dtype)

        data['beam'] = []
        data['beamidx'] = []
        beamidx = self.dataframe['unit1_beam'][index] - 1
        x_data = range(max(beamidx - 5, 0), min(beamidx + 5, 63) + 1)
        #Gaussian distributed target instead of one-hot
        y_data = stats.norm.pdf(x_data, beamidx, 0.5)
        data_beam = np.zeros((64))
        data_beam[x_data] = y_data * 1.25
        data['beam'].append(data_beam)
        data['beamidx'].append(beamidx)

        return data

        

def lidar_to_histogram_features(lidar, address,custom_FoV):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud,addr):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 50
        y_meters_max = 50
        xbins = np.linspace(-x_meters_max, 0, 257)
        ybins = np.linspace(-y_meters_max, y_meters_max, 257)
        # custom field of view of lidar data
        if custom_FoV:
            if 'scenario31' in addr:
                xbins = np.linspace(-70, 0, 257)
                ybins = np.linspace(-25, 14, 257)
            elif 'scenario32' in addr:
                xbins = np.linspace(-60, 0, 257)
                ybins = np.linspace(-40, 5.5, 257)
            elif 'scenario33' in addr:
                xbins = np.linspace(-50, 0, 257)
                ybins = np.linspace(-12, 7, 257)
            elif 'scenario34' in addr:
                xbins = np.linspace(-50, 0, 257)
                ybins = np.linspace(-20, 10, 257)

        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    lidar_feature = splat_points(lidar,address)
    lidar_feature = lidar_feature[np.newaxis, :, :]
    return lidar_feature

def xy_from_latlong(lat_long):
    """
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)


def Normalize_loc(root, dataframe, angle_norm, seq_len=5, root_csv=None):
    n_samples = dataframe.index.stop
    total_instances = 5
    if seq_len < 1 or seq_len > total_instances:
        raise ValueError(
            f"Unsupported seq_len={seq_len}. Expected an integer in [1, {total_instances}] "
            "for the current 5-frame GPS layout."
        )
    selected_indices = list(range(total_instances - seq_len + 1, total_instances + 1))
    required_columns = [f'unit2_loc_{idx}' for idx in selected_indices]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        csv_name = root_csv if root_csv is not None else '<unknown_csv>'
        available_unit2 = [col for col in dataframe.columns if col.startswith('unit2_loc_')]
        raise ValueError(
            f"CSV '{csv_name}' is missing required GPS columns {missing_columns} for seq_len={seq_len}. "
            f"Expected the last {seq_len} GPS slots from unit2_loc_1~unit2_loc_5. "
            f"Available unit2 GPS columns: {available_unit2}"
        )

    pos_rel_paths = [dataframe[col].values for col in required_columns]
    pos_bs_rel_paths = dataframe['unit1_loc'].values
    pos_abs_paths = [[os.path.join(root, path[2:]) for path in rel_paths] for rel_paths in pos_rel_paths]
    pos_bs_abs_paths = [os.path.join(root, path[2:]) for path in pos_bs_rel_paths]
    pos_input = np.zeros((n_samples, seq_len, 2))
    pos_bs = np.zeros((n_samples, 2))
    for sample_idx in tqdm(range(n_samples)):
        # unit2 (UE) positions
        for time_idx, abs_paths in enumerate(pos_abs_paths):
            pos_input[sample_idx, time_idx, :] = np.loadtxt(abs_paths[sample_idx])
        # unit1 (BS) position
        pos_bs[sample_idx] = np.loadtxt(pos_bs_abs_paths[sample_idx])

    pos_ue_stacked = pos_input.reshape(-1, 2)
    pos_bs_stacked = np.repeat(pos_bs, seq_len, axis=0)

    pos_ue_cart = xy_from_latlong(pos_ue_stacked)
    pos_bs_cart = xy_from_latlong(pos_bs_stacked)

    pos_diff = pos_ue_cart - pos_bs_cart

    # pos_min = np.min(pos_diff, axis=0)
    # pos_max = np.max(pos_diff, axis=0)
    pos_max = np.array([40.20955233, 52.31386139])
    pos_min = np.array([ -7.18029715, -97.55563452])

    # Normalize and unstack
    pos_stacked_normalized = (pos_diff - pos_min) / (pos_max - pos_min)
    if angle_norm:
        pos_stacked_normalized = normalize(pos_diff, axis=1)

    pos_input_normalized = pos_stacked_normalized.reshape(n_samples, seq_len, 2)
    if angle_norm:
        angle = np.arctan(pos_input_normalized[..., 1] / pos_input_normalized[..., 0]) / np.pi * 180
        for sample_idx in tqdm(range(n_samples)):
            if 'scenario31' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= -50.52#-40.94#
            if 'scenario32' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= 44.8#39.61#
            if 'scenario33' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= 55.6#47.85#
            if 'scenario34' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= -60#-59.363#
        idx = angle > 90
        angle[idx] -= 180
        idx = angle < -90
        angle[idx] += 180
        angle_rad = angle / 180 * np.pi
        pos_input_normalized[..., 0] = angle_rad
        pos_input_normalized[..., 1] = angle_rad
    return pos_input_normalized
