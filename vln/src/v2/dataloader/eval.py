from .base import BasePathKeyDataloader
import lmdb
import msgpack_numpy
from vln.src.v2.util.eval import(
    generate_eval_key
)

class EvalPathKeyDataloader(BasePathKeyDataloader):
    def __init__(
        self,
        base_data_dir,
        split_data_types,
        robot_offset,
        rank,
        ckpt_name,
        lmdb_path,
        target_scan,
        retry_list,
    ):
        # 加载所有数据
        super().__init__(
            base_data_dir=base_data_dir,
            split_data_types=split_data_types,
            robot_offset=robot_offset,
            filter_same_trajectory=False,
        )
        self.rank = rank
        self.lmdb_path = lmdb_path
        self.target_scan = target_scan
        self.ckpt_name = ckpt_name
        self.retry_list = retry_list
        # 获取当前 rank 需要 eval 的数据
        key = f"eval_rank_{self.rank}"
        database = lmdb.open(f"{self.lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
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
            else:
                target_path_key_list = path_key_list
        # 根据完成情况进行过滤
        filtered_target_path_key_list = []
        for path_key in target_path_key_list:
            eval_key = generate_eval_key(ckpt_name=self.ckpt_name,path_key=path_key)
            with database.begin() as txn:
                value = txn.get(eval_key.encode())
                if value is None:
                    filtered_target_path_key_list.append(path_key)
                else:
                    value = msgpack_numpy.unpackb(value)
                    if value['success'] == 1.0:
                        if 'success' in self.retry_list:
                            filtered_target_path_key_list.append(path_key)
                        else:
                            continue
                    else:
                        fail_reason = value['fail_reason']
                        if fail_reason in retry_list:
                            filtered_target_path_key_list.append(path_key)

        self.eval_path_key_list=filtered_target_path_key_list
        database.close()