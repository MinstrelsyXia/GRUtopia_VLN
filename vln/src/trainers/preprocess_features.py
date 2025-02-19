import os
import lmdb
import torch
import numpy as np
import msgpack_numpy
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from typing import Dict, Any
import copy
import shutil

from vln.src.models.dp_policy_noRNN import CMA_DP_noRNN_Net
from vln.src.models.utils.feature_extract import extract_image_features
from vln.src.models.LongCLIP.model import longclip

class FeaturePreprocessor:
    def __init__(
        self,
        model: CMA_DP_noRNN_Net,
        config: Dict[str, Any],
        train_dataset,
        bert_tokenizer,
        input_lmdb_dir: str,
        output_lmdb_dir: str,
        device: torch.device,
        batch_size: int = 16,
        del_original_rgb: bool = False
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.input_lmdb_dir = input_lmdb_dir 
        self.output_lmdb_dir = output_lmdb_dir
        self.device = device
        self.batch_size = batch_size
        self.del_original_rgb = del_original_rgb
        # Initialize CLIP image processor
        _, self.image_processor = longclip.load(config.MODEL.IMAGE_ENCODER.RGB.model_path)
        self.tokenizer = bert_tokenizer
        
        # Create output directory
        if os.path.exists(output_lmdb_dir):
            shutil.rmtree(output_lmdb_dir)
        os.makedirs(output_lmdb_dir, exist_ok=True)
        print(f"Output LMDB directory: {output_lmdb_dir}")

    def process_batch(self, batch_data):
        """Process a batch of images and extract features"""
        batch_rgb = []
        batch_depth = []
        batch_rgb_lengths = []  # Track lengths of each data item's rgb sequence
        batch_instr = []
        batch_instr_lengths = []

        skip_list = []
        
        # Process each item in batch
        for bs_idx, data in enumerate(batch_data):
            # Get RGB and depth images
            try:
                episode_data = data['episode_data']
            except KeyError:
                skip_list.append(bs_idx)
                batch_rgb_lengths.append(0)
                batch_instr_lengths.append(0)
                continue
            if len(episode_data['camera_info']) == 0:
                skip_list.append(bs_idx)
                batch_rgb_lengths.append(0)
                batch_instr_lengths.append(0)
                continue
            rgb = copy.deepcopy(episode_data['camera_info']['pano_camera_0']['rgb'])
            depth = copy.deepcopy(episode_data['camera_info']['pano_camera_0']['depth'])
            instrs = copy.deepcopy(episode_data['instructions'])
            
            # Store the length of rgb sequence
            batch_rgb_lengths.append(len(rgb))
            batch_instr_lengths.append(len(instrs))
            
            # Process RGB images
            for idx in range(len(rgb)):
                # Convert directly to PIL Image without unnecessary conversions
                image = Image.fromarray(rgb[idx])
                image = self.image_processor(image)
                batch_rgb.append(image)  # Extend instead of append
                batch_depth.append(torch.unsqueeze(torch.from_numpy(depth[idx]), dim=-1))

            # extract instruction tokens
            for instr in instrs:
                token = self.tokenizer(instr)[0]
                batch_instr.append(token)

        # Stack all RGB images into one tensor
        batch_rgb = torch.stack(batch_rgb).to(self.device)  # Now shape is (total_images, C, H, W)
        batch_depth = torch.stack(batch_depth).to(self.device)
        batch_instr = torch.stack(batch_instr).to(self.device)
        
        depth_return_x_before_fc = True if self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'resnet' else False
        
        # Create batch dict
        batch = {
            'mode': 'img_embedding',
            'rgb_inputs': batch_rgb,
            'depth_inputs': batch_depth,
            'img_mod': self.config.MODEL.IMAGE_ENCODER.RGB.img_mod,
            'depth_return_x_before_fc': depth_return_x_before_fc,
            'proj': self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
            'process_images': False  # Images already processed
        }

        # Extract features using model's image encoder
        with torch.no_grad():
            rgb_features, depth_features = self.model(batch)
        
        batch = {
            'mode': 'txt_embedding',
            'instr_inputs': batch_instr,
        }
        with torch.no_grad():
            instr_features = self.model(batch)
            
        # Split rgb_features back according to original lengths
        split_rgb_features = []
        split_depth_features = []
        split_instr_features = []
        start_idx = 0
        adding_idx = 0
        for length in batch_rgb_lengths:
            if adding_idx in skip_list:
                adding_idx += 1
                split_rgb_features.append(None)
                split_depth_features.append(None)
                split_instr_features.append(None)
                continue

            end_idx = start_idx + length
            split_rgb_features.append(rgb_features[start_idx:end_idx])
            split_depth_features.append(depth_features[start_idx:end_idx])
            start_idx = end_idx
            adding_idx += 1

        start_idx = 0
        adding_idx = 0
        for length in batch_instr_lengths:
            if adding_idx in skip_list:
                adding_idx += 1
                split_instr_features.append(None)
                continue
            end_idx = start_idx + length
            split_instr_features.append(instr_features[start_idx:end_idx])
            start_idx = end_idx
            adding_idx += 1
        return split_rgb_features, split_depth_features, split_instr_features

    def preprocess_features(self):
        """Main method to preprocess all features"""
        # Open input LMDB
        with lmdb.open(
            self.input_lmdb_dir,
            map_size=int(1e12),
            readonly=True,
            lock=False
        ) as input_env, \
        lmdb.open(
            self.output_lmdb_dir, 
            map_size=int(1e12),
            writemap=True
        ) as output_env:
            
            # Get all keys
            with input_env.begin(buffers=True) as txn:
                cursor = txn.cursor()
                keys = []
                while cursor.next():
                    key_bytes = bytes(cursor.key())  # Convert memoryview to bytes
                    keys.append(key_bytes.decode('utf-8'))  # Decode bytes to string
            
            # Process in batches
            batch_data = []
            for idx, key in enumerate(tqdm(keys, desc="Processing features")):
                # Get data from input LMDB
                with input_env.begin(buffers=True) as txn:
                    data = msgpack_numpy.unpackb(txn.get(key.encode('utf-8')), raw=False)
                    
                    try:
                        episode_data = data['episode_data']
                    except KeyError:
                        continue
                    if len(episode_data['camera_info']) == 0:
                        continue

                    # add instructions
                    data['episode_data']['instructions'] = []
                    for ep_idx in range(len(self.train_dataset[key])):
                        data['episode_data']['instructions'].append(self.train_dataset[key][ep_idx]['instruction']['instruction_text'][:self.config.MODEL.TEXT_ENCODER.max_length])
                    
                
                batch_data.append(data)
                
                # Process when batch is full or at end
                if len(batch_data) == self.batch_size or idx == len(keys) - 1:
                    rgb_features, depth_features, instr_features = self.process_batch(batch_data)
                    
                    # Save processed features
                    with output_env.begin(write=True) as txn:
                        for i, k in enumerate(keys[idx-len(batch_data)+1:idx+1]):
                            data = copy.deepcopy(batch_data[i])
                            # Update data with new features
                            if rgb_features[i] is not None:
                                data['episode_data']['rgb_features'] = rgb_features[i].cpu().numpy()
                            if depth_features[i] is not None:
                                data['episode_data']['depth_features'] = depth_features[i].cpu().numpy()
                            if instr_features[i] is not None:
                                data['episode_data']['instr_features'] = instr_features[i].cpu().numpy()
                            
                            if self.del_original_rgb:
                                del data['episode_data']['camera_info']['pano_camera_0']['rgb']
                            
                            # Save to output LMDB
                            txn.put(
                                k.encode(),
                                msgpack_numpy.packb(data, use_bin_type=True)
                            )
                    
                    batch_data = []

def main():
    # Initialize model, config etc
    # Call preprocessor
    input_lmdb_dir = "data/sample_episodes/20241207_sample_episodes"
    output_lmdb_dir = "data/sample_episodes/20241207_sample_episodes_processed"
    
    preprocessor = FeaturePreprocessor(
        model=model,
        config=config,
        input_lmdb_dir=input_lmdb_dir,
        output_lmdb_dir=output_lmdb_dir, 
        device=device,
        batch_size=32
    )
    preprocessor.preprocess_features()

if __name__ == "__main__":
    main()
