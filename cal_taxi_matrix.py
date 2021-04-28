import pandas as pd
import numpy as np
import pickle
file_taxi = "F:Taxi-trips/Sep.csv"

num_community = 77
"""num_community = 77
with open('jan.pkl', 'rb') as file:
    arr = pickle.load(file)
print(arr.shape)
print(arr[7][0])
exit(0)"""


def find_index_date_new(x, dic):
    date = x.split()[0]
    value = dic[date]
    # print(value)
    date = pd.to_datetime(date)
    date_range_new = pd.date_range(start=date, freq='4H', periods=7)
    date_range_new = date_range_new.strftime('%Y-%m-%d %#H:%M:%S UTC')
    date_range_new = date_range_new.tolist()
    # print(date_range_new)

    idx_date_new = {}
    for i in range(len(date_range_new)):
        idx_date_new[date_range_new[i]] = i
    # print(idx_date_new)
    date = pd.to_datetime(x)
    for k, v in idx_date_new.items():
        k = pd.to_datetime(k)
        if date <= k:
            # print(date, ",", k)
            # print(value*6+v)
            return value * 6 + v
        # else:
        # print(date, ",", k)
    return -1


taxi = pd.read_csv(file_taxi, low_memory=False)
final_columns = ['trip_start_timestamp', 'trip_end_timestamp', 'pickup_community_area', 'dropoff_community_area']
taxi = taxi[final_columns]
date_range = pd.date_range(start='2019-9-1', end='2019-9-30')
date_range = date_range.strftime('%Y-%m-%d')
date_range = date_range.tolist()
# print(date_range)
idx_date = {}
for i in range(len(date_range)):
    idx_date[date_range[i]] = i
matrix_taxi = np.zeros((num_community, 2, (len(date_range)) * 6), dtype=int)
cnt = 0
exception_cnt = 0
for idx, row in taxi.iterrows():
    cnt = cnt + 1
    if cnt % 50000 == 0:
        print(cnt / 50000, "folds done")
    # print(row)
    try:
        id_region_outflow = int(row['pickup_community_area']) - 1
        id_region_inflow = int(row['dropoff_community_area']) - 1
        id_date_outflow = find_index_date_new(row['trip_start_timestamp'], idx_date)
        id_date_inflow = find_index_date_new(row['trip_end_timestamp'], idx_date)
        # print(id_region_outflow, id_region_inflow, id_date_outflow, id_date_inflow)
        matrix_taxi[id_region_outflow][0][id_date_outflow] += 1
        matrix_taxi[id_region_inflow][1][id_date_inflow] += 1
    except:
        exception_cnt = exception_cnt + 1
        print("exception")

with open('sep.pkl', 'wb') as f:
    pickle.dump(matrix_taxi, f)
# for i in range(num_community):
#    np.savetxt('F:/crime_dataset/Matrices/taxi' + str(i + 1) + '.txt', matrix_taxi[i], fmt='%d')
# print(exception_cnt)
