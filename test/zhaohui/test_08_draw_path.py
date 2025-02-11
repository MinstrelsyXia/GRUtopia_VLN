import lmdb
import msgpack_numpy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

project_path = '/ssd/zhaohui/workspace/w61_grutopia_0107'
lmdb_path_0507 = project_path + '/data/sample_episodes/20250110_dagger/sample_data.lmdb'
env_0508 = lmdb.open(lmdb_path_0507, readonly=True, lock=False)
id=2860
key = f"{id}".encode()
with env_0508.begin() as txn:
    value = txn.get(key)
    value = msgpack_numpy.unpackb(value)
    if value is None:
        print(f"value is None")
        sys.exit()
start_position = [-4.19228983, 12.25430012,  1.150657  ]
goal_position = [-3.29322004,  3.74002004,  1.150657  ]
position_list = value['episode_data']['robot_info']['position']
trajectory = np.array(position_list)

fig, ax = plt.subplots()
ax.plot(trajectory[:, 0], trajectory[:, 1])
plt.plot(start_position[0], start_position[1], 'ro', markersize=6, label='Start')
plt.plot(goal_position[0], goal_position[1], 'bo', markersize=6, label='Goal')
ax.set_title('dagger')
plt.xticks([])
plt.yticks([])
plt.show()
plt.savefig(f"{project_path}/test/zhaohui/map.png", pad_inches=0, bbox_inches='tight', dpi=100)