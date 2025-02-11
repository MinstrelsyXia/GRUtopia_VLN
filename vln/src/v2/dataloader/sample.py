from .base import BasePathKeyDataloader
import lmdb
import msgpack_numpy

class SamplePathKeyDataloader(BasePathKeyDataloader):
    def __init__(
        self,
        base_data_dir,
        split_data_types,
        robot_offset,
        rank,
        lmdb_path,
        target_scan,
        retry_list,
        target_trajectory = None,
        scene_config_file = None,
    ):
        # 加载所有数据
        super().__init__(
            base_data_dir=base_data_dir,
            split_data_types=split_data_types,
            robot_offset=robot_offset,
            filter_same_trajectory=True,
            revise_data=True,
        )
        self.rank = rank
        self.lmdb_path = lmdb_path
        self.target_scan = target_scan
        self.retry_list = retry_list
        self.scene_config_file = scene_config_file
        # 获取当前 rank 需要 eval 的数据
        key = f"sample_rank_{self.rank}"
        database = lmdb.open(f"{self.lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, readonly=True, lock=False)
        with database.begin() as txn:
            value = txn.get(key.encode())
            if value is None:
                desc = f"获取 eval 列表失败,[key:{key}][lmdb_path:{self.lmdb_path}]"
                raise Exception(desc)
            self.eval_map = msgpack_numpy.unpackb(value)
        
        target_path_key_list=[]
        for scan, path_key_list in self.eval_map.items():
            if scan != target_scan:
                continue
            if target_trajectory is None:
                target_path_key_list = path_key_list
            else:
                for path_key in path_key_list:
                    trajectory_id = path_key.split('_')[0]
                    if trajectory_id == str(target_trajectory):
                        target_path_key_list = [path_key]
                        break
        # TODO 根据完成情况进行过滤
        filtered_target_path_key_list = []
        for path_key in target_path_key_list:
            trajectory_id = path_key.split('_')[0]
            sample_key = str(trajectory_id)
            with database.begin() as txn:
                value = txn.get(sample_key.encode())
                if value is None:
                    filtered_target_path_key_list.append(path_key)
                else:
                    value = msgpack_numpy.unpackb(value)
                    if value['finish_status'] == 'success':
                        if 'success' in self.retry_list:
                            filtered_target_path_key_list.append(path_key)
                        else:
                            continue
                    else:
                        fail_reason = value['fail_reason']
                        if fail_reason in retry_list:
                            filtered_target_path_key_list.append(path_key)

        self.sample_path_key_list=filtered_target_path_key_list
        database.close()