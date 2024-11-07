from vlmaps.vlmaps.utils.llm_utils import parse_spatial_instruction

import os
import json
import httpx
import httpcore

print(f"httpx version: {httpx.__version__}")
print(f"httpcore version: {httpcore.__version__}")

def read_instructions_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"Instructions read from {file_path}")
    return data

# 使用示例
file_path = 'tmp/instructions.json'
instructions = read_instructions_from_file(file_path)

text=parse_spatial_instruction(instructions[0]['instruction'])
print(text)