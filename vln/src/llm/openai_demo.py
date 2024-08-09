import base64
import requests

# OpenAI API Key
api_key = "sk-uxafF08zHYhVtHbw570c530d266847B7AfFcF04cBc5b60F3"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
# image_path = "/home/wangliuyi/code/w61-grutopia/logs/images/val_seen/251/occupancy_100.jpg"

test_prompt = '''I need you to guide a robot to follow the given instructions to navigate indoor. I would provide you an image with full semantic information, and a top-down map with the robot's current position, and its ego-centric observation. You need to first describe the robot's current state, and inference the most possible next waypoints based on the given information. You can output the possible coordinates, or the possible region on the map. If you think the robot should turn around to get more information, you can also output the order to let robot explore more.
The given instruction is: Exit the bedroom, cross the room to the left and stop near the room divider on the left.
The given visual semantic image is given.'''
image_path = "/home/wangliuyi/code/w61-grutopia/vln/src/llm/test_imgs/5593/BuildMap_20240808v1.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          # "text": "What’s in this image?"
          "text": test_prompt
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 500
}

response = requests.post("https://api.openai120.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())