import os.path as osp
import pickle as pkl
import sys
import pandas as pd
import numpy as np
import torch
from cogdl.data import Dataset, Graph
from cogdl.utils import remove_self_loops, download_url, untar, coalesce, MAE, CrossEntropyLoss
import os
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import geopy.distance # to compute distances between stations
import glob
from tqdm import tqdm
import warnings
from numpy.core.umath_tests import inner1d

def files_exist(files):
    return all([osp.exists(f) for f in files])


def PeMS_lane_columns(x):
    tmp = [
            'lane_N_samples_{}'.format(x),
            'lane_N_flow_{}'.format(x),
            'lane_N_avg_occ_{}'.format(x),
            'lane_N_avg_speed_{}'.format(x),
            'lane_N_observed_{}'.format(x)
            ]
    return tmp

def raw_data_processByNumNodes(raw_dir, num_nodes, meta_file_name):

    PeMS_daily = os.path.join(f'{raw_dir}', '*')
    PeMS_metadata = os.path.join(f'{raw_dir}', meta_file_name)
    output_dir = os.path.join(f'{raw_dir}')

    # Parameters
    outcome_var = 'avg_speed'
    files = glob.glob(PeMS_daily)
    files.remove(glob.glob(PeMS_metadata)[0])
    PeMS_columns = ['timestamp', 'station', 'district', 'freeway_num',
                    'direction_travel', 'lane_type', 'station_length',
                    'samples', 'perc_observed', 'total_flow', 'avg_occupancy',
                    'avg_speed']
    #PeMS_lane_columns = lambda x: ['lane_N_samples_{}'.format(x),
    #                               'lane_N_flow_{}'.format(x),
    #                               'lane_N_avg_occ_{}'.format(x),
    #                               'lane_N_avg_speed_{}'.format(x),
    #                               'lane_N_observed_{}'.format(x)]
    
    PeMS_all_columns = PeMS_columns.copy()
    for i in range(1, 9):
        PeMS_all_columns += PeMS_lane_columns(i)

    # Randomly select stations to build the dataset
    np.random.seed(42)
    station_file = files[0]
    station_file_content = pd.read_csv(station_file, header=0, names=PeMS_all_columns)
    station_file_content = station_file_content[PeMS_columns]
    station_file_content = station_file_content.dropna(subset=[outcome_var])
    unique_stations = station_file_content['station'].unique()
    selected_stations = np.random.choice(unique_stations, size=num_nodes, replace=False)

    # Build two-months of data for the selected stations/nodes
    station_data = pd.DataFrame({col: []} for col in PeMS_columns)
    for station_file in tqdm(files):
        # Get file date
        file_date_str = station_file.split(os.path.sep)[-1].split('.')[0]
        file_date = datetime(int(file_date_str.split('_')[-3]), int(file_date_str.split('_')[-2]),
                             int(file_date_str.split('_')[-1]))
        # Check if weekday
        if file_date.weekday() < 5:
            # Read CSV
            station_file_content = pd.read_csv(
                station_file, header=0, names=PeMS_all_columns)
            # Keep only columns of interest
            station_file_content = station_file_content[PeMS_columns]
            # Keep stations
            station_file_content = station_file_content[
                station_file_content['station'].isin(selected_stations)]
            # Append to dataset
            station_data = pd.concat([station_data, station_file_content])
    # Drop the 11 rows with missing values
    station_data = station_data.dropna(subset=['timestamp', outcome_var])
    station_data.head()
    station_data.shape
    station_metadata = pd.read_table(PeMS_metadata)
    station_metadata = station_metadata[['ID', 'Latitude', 'Longitude']]
    # Filter for selected stations
    station_metadata = station_metadata[station_metadata['ID'].isin(selected_stations)]
    station_metadata.head()
    # Keep only the required columns (time interval, station ID and the outcome variable)
    station_data = station_data[['timestamp', 'station', outcome_var]]
    station_data[outcome_var] = pd.to_numeric(station_data[outcome_var])
    # Reshape the dataset and aggregate the traffic speeds in each time interval
    V = station_data.pivot_table(index=['timestamp'], columns=['station'], values=outcome_var, aggfunc='mean')
    V.head()
    V.shape
    # Compute distances
    distances = pd.crosstab(station_metadata.ID, station_metadata.ID, normalize=True)
    distances_std = []
    for station_i in selected_stations:
        for station_j in selected_stations:
            if station_i == station_j:
                distances.at[station_j, station_i] = 0
            else:
                # Compute distance between stations
                station_i_meta = station_metadata[station_metadata['ID'] == station_i]
                station_j_meta = station_metadata[station_metadata['ID'] == station_j]
                if np.isnan(station_i_meta['Latitude'].values[0]) or np.isnan(station_i_meta['Longitude'].values[0]) or np.isnan(station_j_meta['Latitude'].values[0]) or np.isnan(station_j_meta['Longitude'].values[0]):
                    d_ij = 0
                else:
                    d_ij = geopy.distance.geodesic(
                        (station_i_meta['Latitude'].values[0], station_i_meta['Longitude'].values[0]),
                        (station_j_meta['Latitude'].values[0], station_j_meta['Longitude'].values[0])).m
                distances.at[station_j, station_i] = d_ij
                distances_std.append(d_ij)
    distances_std = np.std(distances_std)
    distances.head()
    W = pd.crosstab(station_metadata.ID, station_metadata.ID, normalize=True)
    epsilon = 0.1
    sigma = distances_std
    for station_i in selected_stations:
        for station_j in selected_stations:
            if station_i == station_j:
                W.at[station_j, station_i] = 0
            else:
                # Compute distance between stations
                d_ij = distances.loc[station_j, station_i]
                # Compute weight w_ij
                w_ij = np.exp(-d_ij ** 2 / sigma ** 2)
                if w_ij >= epsilon:
                    W.at[station_j, station_i] = w_ij
    W.head()
    # Save to file
    V = V.fillna(V.mean())
    V.to_csv(os.path.join(output_dir, 'V_{}.csv'.format(num_nodes)), index=True)
    W.to_csv(os.path.join(output_dir, 'W_{}.csv'.format(num_nodes)), index=False)
    station_metadata.to_csv(os.path.join(output_dir, 'station_meta_{}.csv'.format(num_nodes)), index=False)


def read_stgcn_data(folder, num_nodes):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    W = pd.read_csv(osp.join(folder, "W_{}.csv".format(num_nodes)))
    T_V = pd.read_csv(osp.join(folder, "V_{}.csv".format(num_nodes)))
    V = T_V.drop('timestamp',axis=1)
    num_samples, num_nodes = V.shape
    scaler = StandardScaler()


    # format graph for pyg layer inputs
    G = sp.coo_matrix(W)
    edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
    edge_weight = torch.tensor(G.data).float().to(device)
    data = Graph()
    data.num_nodes = num_nodes
    data.num_samples = num_samples
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    data.scaler = scaler
    data.V = V
    data.W = W
    data.timestamp = T_V['timestamp']
    data.node_ids = V.columns

    return data


class STGCNDataset(Dataset):
    def __init__(self, root, name, num_stations, meta_file_name):
        self.name = name
        self.meta_file_name = meta_file_name
        self.url = "https://cloud.tsinghua.edu.cn/f/5af7ea1a7d064c5ba6c8/?dl=1"
        self.url_test = "https://cloud.tsinghua.edu.cn/f/a39effe167df447eab80/?dl=1"
        self.num_stations = num_stations
        super(STGCNDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])
        self.num_nodes = self.data.num_nodes

    @property
    def raw_file_names(self):
        names = ["station_meta_{}.csv".format(self.num_stations), "V_{}.csv".format(self.num_stations), "W_{}.csv".format(self.num_stations)]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        # if os.path.exists(self.raw_dir+r'\PeMS_20210501_20210630'):  # pragma: no cover
        
        # TODO: Auto Traffic pipeline support
        # if os.path.exists(self.raw_dir):  # auto_traffic
        #     return
        
        _test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_path = _test_path[:(len(_test_path)-len("/cogdl"))]+"/tests/test_stgcn/"
        if os.path.exists(test_path):
            download_url(self.url_test, self.raw_dir, name=self.name + ".zip")
        else:
            download_url(self.url, self.raw_dir, name=self.name + ".zip")
        untar(self.raw_dir, self.name + ".zip")

    def process(self):
        files = self.raw_paths
        if not files_exist(files):
            raw_data_processByNumNodes(self.raw_dir, self.num_stations, self.meta_file_name)
        data = read_stgcn_data(self.raw_dir, self.num_stations)
        torch.save(data, self.processed_paths[0])


    def __repr__(self):
        return "{}".format(self.name)


    def get_evaluator(self):
        return MAE()

    def get_loss_fn(self):
        return torch.nn.MSELoss()


class PeMS_Dataset(STGCNDataset):
    def __init__(self, data_path="data"):
        dataset = "pems-stgcn"
        root = osp.join(data_path, dataset)
        super(PeMS_Dataset, self).__init__(root, dataset, num_stations=288, meta_file_name= 'd07_text_meta.txt')
