import sys
sys.path.append("./")

import sys, os
add_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(add_path)

sys.path.append(r'C:\Users\user\Desktop\Aotu_cogdl\cogdl-trafficPre-master')

import sys, os
add_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(add_path)

from cogdl import utils
from examples import experiment
import os.path as osp
import os
import numpy as np
import gzip
import time
import requests
from bs4 import BeautifulSoup
import re
from lxml import etree
import json
import shutil
import pandas as pd
import os
import datetime as dt
import numpy as np
import datetime
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
import streamlit.components.v1 as components
import flickrapi
import random
from dotenv import load_dotenv
import os
import urllib
import numpy as np
import pandas as pd
import folium
from cogdl.utils import makedirs
import tqdm


def get_timestamp(date):
    return datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp()


def get_location(ID):
    return [id2Location[ID]['lat'], id2Location[ID]['lon']]


def process_meta(df_meta):
    id2Location = {}
    ID = df_meta["ID"]
    lat = df_meta["Latitude"]
    lon = df_meta["Longitude"]
    for index, i in enumerate(ID):
        id2Location[i] = {'lat': lat[index], 'lon': lon[index]}
    return id2Location


def un_gz(file_name):
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


def files_exist(files):
    return all([osp.exists(f) for f in files])


def download_url(sss, url, folder, name=None, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        name (string): saved filename.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    if log:
        print("Downloading", url)

    makedirs(folder)

    try:
        data = sss.get(url, stream=True)
        total = int(data.headers.get('content-length', 0))
    except Exception as e:
        print("Failed to download the dataset.")
        print(f"Please download the dataset manually and put it under {folder}.")
        exit(1)

    if name is None:
        filename = url.rpartition("/")[2]
    else:
        filename = name
    path = osp.join(folder, filename)

    with open(path, "wb") as f:
        for da in data.iter_content(chunk_size=1024):
            size = f.write(da)

    return path


if __name__ == "__main__":

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import warnings
    warnings.filterwarnings("ignore")

    kwargs = {"epochs": 1,
              "kernel_size": 3,
              "n_his": 20,
              "n_pred": 1,
              "channel_size_list": np.array([[1, 16, 64], [64, 64, 64], [64, 16, 64]]),
              "num_layers": 3,
              "num_nodes": 288,
              "train_prop": 0.8,
              "val_prop": 0.1,
              "test_prop": 0.1,
              "pred_length": 288,
              }


    meta_path = os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/raw/station_meta_288.csv'
    raw_path = os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/raw'
    process_path = os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/processed'
    latest_data_name = "None"


    current_year = datetime.datetime.now().year
    s = requests.Session()
    username = "your mail"
    password = "your password"
    login_url = "https://pems.dot.ca.gov"
    # search_url= "https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=7&submit=Submit"
    search_url = f"https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy={current_year}&type=station_5min&returnformat=text"
    url_meta = f"https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy={current_year}&type=meta&returnformat=text"
    result = s.post(url=login_url, data={"username": username, "password": password, "commit": "login"})
    page = s.get(search_url)
    meta_page = s.get(url_meta)
    soup = BeautifulSoup(page.content, "lxml")
    soup_meta = BeautifulSoup(meta_page.content, "lxml")
    list_url = soup.text
    meta_list_url = soup_meta.text


    result_json = json.loads(list_url)
    meta_result_json = json.loads(meta_list_url)
    data_month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                    "November", "December"]
    data_month_num2str = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 7:"July", 8:"August", 9:"September", 10:"October",
                    11:"November", 12:"December"}
    detail = "detail"


    # """
    # {"data":
    #     {
    #     "October":[{"file_name":"d07_text_meta_2021_10_20.txt","file_id":"435257","bytes":"411,747","url":"\/?download=435257&dnode=Clearinghouse"}],
    #     "March":[{"file_name":"d07_text_meta_2022_03_12.txt","file_id":"442594","bytes":"412,077","url":"\/?download=442594&dnode=Clearinghouse"}],
    #     "July":[{"file_name":"d07_text_meta_2022_07_15.txt","file_id":"443890","bytes":"411,315","url":"\/?download=443890&dnode=Clearinghouse"}]
    #     },
    #     "detail":{"district":"7","month":7,"year":2022,"date":"2022","data_set":"meta"}
    #     }
    # """

    # """
    # {"file_name":"d07_text_station_5min_2021_01_01.txt.gz","file_id":"409509","bytes":"29,673,928","url":"\/?download=409509&dnode=Clearinghouse"}
    # "detail":{"district":"7","month":12,"year":2021,"date":"2021","data_set":"station_5min"}}
    # """


    latest_data_month = data_month_num2str[result_json['detail']["month"]]
    latest_data_year = result_json['detail']["year"]
    latest_data_url = ""
    latest_data_url = 'https://pems.dot.ca.gov' + result_json['data'][latest_data_month][-1]["url"]
    latest_data_name = result_json['data'][latest_data_month][-1]["file_name"]
    latest_data = result_json['data'][latest_data_month][-1]["file_name"].split('.')[0].split('5min_')[1]

    # latest_meat_data_month = data_month_num2str[meta_result_json['detail']["month"]]
    latest_meat_data_month = data_month_num2str[3]
    latest_meta_data_year = meta_result_json['detail']["year"]
    latest_meta_data_url = ""
    latest_meta_data_url = 'https://pems.dot.ca.gov' + meta_result_json['data'][latest_meat_data_month][-1]["url"]
    latest_meta_data_name = meta_result_json['data'][latest_meat_data_month][-1]["file_name"]
    latest_meta_data = meta_result_json['data'][latest_meat_data_month][-1]["file_name"].split('.')[0].split('meta_')[1]

    # """
    # https://pems.dot.ca.gov/?download=430913&amp;dnode;=Clearinghouse
    # https://pems.dot.ca.gov/   +   ?download=442594&dnode=Clearinghouse
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=station_5min&returnformat=text
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=station_hour&returnformat=text
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=station_day&returnformat=text
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=meta&returnformat=text
    # """


    # latest_data_url = "https://down.wss.show/mt36wnf/8/yo/8yoxmt36wnf?cdn_sign=1659625360-66-0-35778ab0c7692c94bf2459add6c866a5&exp=1200&response-content-disposition=attachment%3B%20filename%3D%22d07_text_station_5min_2022_07_31.txt.gz%22%3B%20filename%2A%3Dutf-8%27%27d07_text_station_5min_2022_07_31.txt.gz"
    # meta_data_url = "https://down.wss.show/tftoq18/8/yq/8yqrtftoq18?cdn_sign=1659626090-80-0-ed58eb2d57251c3ce7102fddce0583d7&exp=240&response-content-disposition=attachment%3B%20filename%3D%22d07_text_meta_2022_03_12.txt%22%3B%20filename%2A%3Dutf-8%27%27d07_text_meta_2022_03_12.txt"

    if osp.exists(raw_path+'/'+latest_data_name):
        print("==============================no upadta================================")
        pass
    else:
        print(raw_path)
        if osp.exists(raw_path):
            shutil.rmtree(raw_path)
            makedirs(raw_path)
        else:
            makedirs(raw_path)
        if osp.exists(process_path):
            shutil.rmtree(process_path)
        download_url(s, latest_data_url, raw_path, name=latest_data_name)
        download_url(s, latest_meta_data_url, raw_path, name="d07_text_meta.txt")
        experiment(dataset="pems-50", model="stgcn", resume_training=True,  devices=[1], **kwargs)



    pre_V_path = os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/stgcn_prediction.csv'
    meta = pd.read_csv(meta_path)
    pre_V = pd.read_csv(pre_V_path)

    id2Location = process_meta(meta)
    time_timestamp = list(pre_V['timestamp'])
    demo_data_ID = []
    demo_data_time = []
    demo_data_lat = []
    demo_data_lon = []
    demo_data_pre_speed = []
    for i in meta["ID"]:
        for index, j in enumerate(pre_V[str(i)]):
            demo_data_ID.append(i)
            s = time_timestamp[index]
            a = dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
            b = dt.datetime.strftime(a, '%Y-%m-%d %H:%M:%S')
            demo_data_time.append(b)
            demo_data_lat.append(get_location(i)[0])
            demo_data_lon.append(get_location(i)[1])
            demo_data_pre_speed.append(j)
    demo_data = pd.DataFrame()
    demo_data["ID"] = demo_data_ID
    demo_data["timestamp"] = demo_data_time
    demo_data["Latitude"] = demo_data_lat
    demo_data["Longitude"] = demo_data_lon
    demo_data["predict_speed"] = demo_data_pre_speed
    demo_data.to_csv(os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/demo_data.csv', index=False)
    travel_log = demo_data
    timestamp = travel_log['timestamp']
    timestamp_set = set(timestamp)
    new_timeatamp = list(timestamp_set)

    new_sort_timeatamp = sorted(new_timeatamp, key=lambda date: get_timestamp(date))

    data_move = []
    for i in new_sort_timeatamp:
        df = travel_log[(travel_log['timestamp'] == i)]
        num = df.shape[0]
        lat = np.array(df["Latitude"][0:num])
        lon = np.array(df["Longitude"][0:num])
        speed = np.array(df["predict_speed"][0:num], dtype=float)
        _data = [[lat[i], lon[i], speed[i]] for i in range(num)]
        data_move.append(_data)

    load_dotenv()

    # set page layout
    st.set_page_config(
        page_title="PeMS traffic predict by CogDl",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    @st.cache
    def load_data():
        """ Load the cleaned data with latitudes, longitudes & timestamps """
        travel_log = pd.read_csv(
            '/home/renxiangsheng/Auto_trafficPre/examples/simple_stgcn/data/pems-stgcn/demo_data.csv')
        # travel_log = pd.read_csv("/home/xiangsheng/zp/cogdl-trafficPre/examples/simple_stgcn/data/pems-stgcn/demo_streamlit.csv")
        # travel_log = pd.read_csv("./data/pems-stgcn/demo_streamlit.csv")
        travel_log["date"] = pd.to_datetime(travel_log["timestamp"])
        return travel_log


    st.title("🌍 PeMS traffic predict by CogDl")

    travel_data = load_data()
    map_osm = folium.Map(location=[34, -118], zoom_start=10, tiles='Stamen Toner', control_scale=True)

    hm = folium.plugins.HeatMapWithTime(data_move, index=new_sort_timeatamp, name="Speed Map", radius=15,
                                        min_opacity=0.5,
                                        max_opacity=0.6, scale_radius=False,
                                        gradient={.1: 'blue', .2: 'lime', .3: 'red', .4: 'black', .5: 'pink',
                                                    .6: 'green', 1: 'yellow'},
                                        use_local_extrema=True, auto_play=False,
                                        display_index=True, index_steps=1, min_speed=1,
                                        max_speed=100, speed_step=1, position='bottomleft',
                                        overlay=True, control=True, show=True)
    hm.add_to(map_osm)
    st.subheader("Heatmap")
    fig = folium.Figure().add_child(map_osm)
    components.html(fig.render(), height=800, width=1600)
    file_path = "./demo_traffic.html"
    hm.save(file_path)

    print("--------------------------------------------------------waiting tomorrow data update---------------------------------------------------------------")
    time.sleep(3600*24)

    st.experimental_rerun()
