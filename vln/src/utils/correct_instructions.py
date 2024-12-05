from pathlib import Path
import json
from tqdm.notebook import tqdm
import os
import openai
data_folder = Path("/isaac-sim/GRUtopia/data/datasets/revised/statistics")
inst_alias = "inst"
splits = ["train", "val_seen","val_unseen"]
action_map = {
    0: "stop",
    1: "move forward",
    2: "turn left",
    3: "turn right",
}
for split in splits:
    data = {}
    split_folder = data_folder/split
    for ep_folder in tqdm(list(split_folder.glob("*"))):
        for inst_path in ep_folder.glob("inst/*.txt"):
            with open(inst_path,"r") as f:
                inst = f.read()
            info_path = Path(str(inst_path).replace(".txt", ".info"))
            with open(info_path, "r") as f:
                info = json.load(f)
            action_path = ep_folder / "action" / "0.json"
            with open(action_path, "r") as f:
                action = json.load(f)
            action = [action_map[v] for v in action]
            k = 1
            while action[k]==action[0]:
                k += 1
            traj_id = ep_folder.stem
            data[traj_id+"_"+inst_path.stem] = {
                "gt_first_action": action[0],
                "gt_second_action": action[k],
                "llm_inst_first_action": info["actions_compact"]["0"],
                "llm_inst_second_action": info["actions_compact"]["1"] if "1" in info["actions_compact"] else "",
                "instruction": inst,
            }
        # print(data)
        # break
    with open("{}_first_action.json".format(split), "w") as f:
        json.dump(data, f, indent=2)
    errors = []
    for k,v in data.items():
        gt = v["gt_first_action"].lower()
        llm = v["llm_inst_first_action"].lower()
        if "right" in gt and "left" in llm:
            errors.append(k)
        elif "right" in llm and "left" in gt:
            errors.append(k)
    print("Error instructions in {}: {}".format(split, len(errors)))