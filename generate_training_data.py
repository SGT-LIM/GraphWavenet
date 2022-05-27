from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, que_len_df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, weather = False, que_len=False, scaler=None #weather_df
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    #df=df.set_index("STATISTICSDATE")

    num_samples, num_nodes = df.shape
    print(num_samples, num_nodes)
    num_nodes = num_nodes-1 #'STATISTICSDATE'를 index로 변환 전, STATISTICSDATE 시간 데이터까지 포함돼서 215개라서.. 하나 빼주기.
    df.set_index('OCRN_DT', append=False, inplace=True) #인덱스로 설정 #STATISTICSDATE

    #num_samples = num_samples-4 #날씨데이터랑 크기 맞춰주려고 하나 빼는 것. + window 맞춰주려고.
    #df = df.iloc[3:-1] #날씨데이터랑 크기 맞춰주려고 하나 빼는 것. + window 맞춰주려고.
    print(df.iloc[:2])

    data = np.expand_dims(df.values, axis=-1) #차원 하나 추가. data.shape은 (2400,214,1)이 됨. 
    print(data.shape)

    feature_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values.astype("datetime64[s]") - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")#시간을 second단위로 하고 하루=86400초로 나눔.
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))#data랑 같은 shape이 됨.(2400,214,1)
        feature_list.append(time_in_day)

    if add_day_in_week:
        df.index = df.index.values.astype("datetime64[s]") #원래는 object 타입이라 이렇게 시간 타입으로 변환해줘야 함.
        dow = df.index.dayofweek # 월요일은 0, 화요일은 1, 수요일은 2, ...으로 변환.
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
    """
    if weather:
        #weather_df = weather_df.iloc[1:]
        #temp = weather_df['기온(°C)']
        #temp_tiled = np.tile(temp, [1, num_nodes, 1]).transpose((2, 1, 0))
        #feature_list.append(temp_tiled)

        rain = weather_df['강수량(mm)']
        rain_tiled = np.tile(rain, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(rain_tiled)

        snow = weather_df['적설(cm)']
        snow_tiled = np.tile(snow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(snow_tiled)
    """
    if que_len:
        que_len_df.set_index('TOT_DT', append=False, inplace=True) #인덱스로 설정 #원래는 TOT_DT말고 STATISTICSDATE
        print(que_len_df)
        que_data = np.expand_dims(que_len_df.values, axis=-1)
        print(que_data.shape)
        feature_list.append(que_data)

    data = np.concatenate(feature_list, axis=-1) #feature_list에 array로서 각각 있던 data와 time_in_day를 합쳐줌. (2400, 214, 2)가 됨. 
    #[[[21.  0.19791667]
    # [57. 0.19791667]
    # ...
    # [39. 0.19791667]] #여기부터 위가 쭉 동시간대의 214개 노드의 각각 속도.

    # [[28. 0.20138889]
    # ...
    # [39. 0.20138889]]
    # ...
    # ]
    print(data)
    print(data.shape)

    x, y = [], []
    min_t = abs(min(x_offsets)) #11
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive    # 2400-12 = 2388
    print(min_t, max_t)
    for t in range(min_t, max_t):  # t is the index of the last observation. #t = 11 ~ 2387까지. 총 2377번 반복.
        x.append(data[t + x_offsets, ...]) # t=11일 때: data[[0,1,2,...,11], ...]이고, 이건 data에서 맨앞에서부터 12개의 시간 각각에 따른 214개 노드의 속도 데이터임. shape는 (12,214,2).
        y.append(data[t + y_offsets, ...]) # t=11일 때: data[[12,13,14,...,23], ...]이고, 이건 data에서 12번째 시간에서부터 12개의 시간 각각에 따른 214개 노드의 속도 데이터임. shape는 (12,214,2).
    x = np.stack(x, axis=0) # x.shape = (2377,12,214,2)
    y = np.stack(y, axis=0) # y.shape = (2377,12,214,2)
    print(x.shape)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_csv(args.traffic_df_filename, low_memory=False)
    #weather_df = pd.read_csv(args.weather_filename, low_memory=False) #날씨 데이터 추가한 것.
    que_len_df = pd.read_csv(args.que_length_filename, low_memory=False) #대기행렬 데이터 추가한 것.

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),))) #array([-11,-10,...,0])
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1)) #array([1,2,...12])
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df, que_len_df,        #날씨 데이터, 대기행렬 데이터 추가한 것. #weather_df
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True, #원래는 True
        add_day_in_week=False,
        weather = False,
        que_len=False
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0] #2377개. 가능한 총 window의 개수??
    num_test = round(num_samples * 0.2) 
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train] #x_train.shape = (1664, 12, 214, 2)
    
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    print(x_test.shape)
    
    
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/2022년_부천시_대기행렬예측", help="Output directory.")
    #parser.add_argument("--traffic_df_filename", type=str, default="data/extracted_links_and_velocity_to34272_fillna60.csv", help="Raw traffic readings.",) 
    parser.add_argument("--traffic_df_filename", type=str, default="data/30s단위_대기행렬.csv", help="Raw traffic readings.",) #원래는 vel572_FromJan이어야 하나 지금은 대기행렬을 예측해야하므로 바꿔줌.
    #parser.add_argument("--weather_filename", type=str, default="data/부천시_날씨데이터_scaled_tma없음.csv", help="weather data readings.",)
    parser.add_argument("--que_length_filename", type=str, default="data/vel572_FromJan.csv", help="que_length data readings.",)
    #parser.add_argument("--traffic_df_filename", type=str, default="data/temperature_oiloxy_100000.csv", help="Raw traffic readings.")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
