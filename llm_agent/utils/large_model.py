import os
import sys
import io
import re
import json
import base64
import torch
import tiktoken
import requests
import dashscope
from PIL import Image
from http import HTTPStatus
from openai import AzureOpenAI, OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT_DIR, 'llm_agent/agent/vlm_prompt.txt'), 'r', encoding='utf-8') as file:
    vlm_prompt = file.read()
######################## llm ########################
class Llm_intern_local:
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.url = 'http://10.6.8.58:5000/process'
        # self.llm = {'tokenizer': AutoTokenizer.from_pretrained("internlm/internlm2-chat-1_8b", trust_remote_code=True), 'model': AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()}
    def get_answer(self, template_type, **kwargs):
        prompt = generate_prompt(template_type, **kwargs)
        question = {
            'prompt': prompt,
            'history': []
        }
        answer = requests.post(self.url, json=question)
        if answer.status_code == 200:
            answer_data = answer.json()
            answer_data = answer_data.get('data')
            if answer_data:
                result = answer_data.get('result')
        else:
            print("Failed to retrieve data. Status Code:", answer.status_code)
        # result, self.history = self.llm['model'].chat(self.llm['tokenizer'], prompt, self.history)
        if self.verbose:
            print('prompt:\n', prompt)
            print('result:\n', result)
        return result
    
class Llm_intern:
    def __init__(self, verbose = False, try_times=20):
        self.verbose = verbose
        self.try_times = try_times
        with open('api_key/internlm_api.txt', 'r', encoding='utf-8') as file:
            api_key = file.read().strip()
        self.url = 'https://puyu.openxlab.org.cn/puyu/api/v1/chat/completions'
        self.header = {'Content-Type': 'application/json', 
                       "Authorization": "Bearer "+api_key}
        
    def get_answer(self, template_type, **kwargs):
        prompt = generate_prompt(template_type, **kwargs)
        data = {
                "model": "internlm2-latest",  
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "n": 1,
                "temperature": 0.8,
                "top_p": 0.9
            }
        
        cnt = 0
        while cnt < self.try_times:
            try:
                response = requests.post(self.url, headers=self.header, data=json.dumps(data))
                result = response.json()["choices"][0]["message"]["content"]
                break
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        if self.verbose:
            print('prompt:', prompt)
            print('result:', result)
        return result
    
class Llm_aliyun:
    def __init__(self, model_name = 'chatglm3-6b', verbose = False, try_times=20):
        self.verbose = verbose
        self.try_times = try_times
        dashscope.api_key_file_path='api_key/aliyun_key.txt'
        self.model_name = model_name
        
    def get_answer(self, template_type, **kwargs):
        prompt = generate_prompt(template_type, **kwargs)
        messages = [
        {'role': 'system', 'content':'You never explain your answer and always keep your answers concise. Remember answering in English.'}
        ,{'role': 'user', 'content': prompt}]

        cnt = 0
        while cnt < self.try_times:
            try:
                response = dashscope.Generation().call(self.model_name, messages=messages, result_format='message')
                if response.status_code == HTTPStatus.OK:
                    result = response.output.choices[0].message.content
                    break
                else:
                    continue
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        if self.verbose:
            print('prompt:', prompt)
            print('result:', result)
        return result
    
######################## vlm ########################
class Vlm_llava:
    def __init__(self, verbose = False, try_times=20):
        self.verbose = verbose
        self.url = 'http://10.6.8.58:5000/process'

    def get_answer(self, template_type, **kwargs):
        image_buf = kwargs['image']
        cnt = 0
        prompt = generate_prompt(template_type, **kwargs)
        img_encoded = base64.b64encode(image_buf.getvalue()).decode('utf-8')
        image_buf.close()
        question = {
            'prompt': prompt,
            'image': img_encoded
        }
        cnt = 0
        while cnt < self.try_times:
            try:
                response = requests.post(self.url, json=question)
                if response.status_code == HTTPStatus.OK:
                    result = response.json().get('data').get('result')
                    break
                else:
                    continue
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        if self.verbose:
            print('prompt:', prompt)
            print('result:', result)
        return result

class Vlm_qwen:
    def __init__(self, verbose = False, try_times=20):
        self.verbose = verbose
        self.try_times = try_times
        dashscope.api_key_file_path='api_key/aliyun_key.txt'
        
    def get_answer(self, template_type, **kwargs):
        image_buf = kwargs['image']
        cnt = 0
        prompt = generate_prompt(template_type, **kwargs)
        Image.open(image_buf).save('vlm_image.png')
        img_path = os.path.abspath('vlm_image.png')
        image_buf.close()
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{img_path}"},
                    {"text": prompt + ' Please answer in English.'}
                ]
            }
        ]
        cnt = 0
        while cnt < self.try_times:
            try:
                response = dashscope.MultiModalConversation.call(model='qwen-vl-plus', messages=messages, stream=False)
                if response.status_code == HTTPStatus.OK:
                    result = response.output.choices[0].message.content[0]['text']
                    break
                else:
                    continue
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        if self.verbose:
            print('prompt:', prompt)
            print('result:', result)
        return result
    
class Vlm_intern:
    def __init__(self, verbose = False, try_times=20):
        self.verbose = verbose
        self.try_times = try_times
    
    def get_answer(self, template_type, **kwargs):
        image_buf = kwargs['image']
        cnt = 0
        prompt = generate_prompt(template_type, **kwargs)
        img_encoded = base64.b64encode(image_buf.getvalue()).decode('utf-8')
        image_buf.close()

        data = {"question": "prompt",
        "image": img_encoded,
        "temperature": 0.,
        "max_num": 12, # 控制图像分辨率，【6，12，18，24】，24 OCR最强
        "max_new_tokens": 1024,
        "do_sample": False,
        }

        while cnt < self.try_times:
            try:
                response = requests.post("http://112.111.7.64:10067/internvl_1_5/chat", json=data)
                result = response.json().get('answer')
                break
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        if self.verbose:
            print('prompt:', prompt)
            print('result:', result)
        return result
        
class Vlm_gpt4o:
    def __init__(self, azure = False, verbose = False, try_times=20):
        if azure:
            with open(os.path.join(ROOT_DIR, 'api_key/azure_api_key.txt'), 'r', encoding='utf-8') as file:
                api_key = file.read().strip()
            with open(os.path.join(ROOT_DIR, 'api_key/azure_api_key_e.txt'), 'r', encoding='utf-8') as file:
                api_key_e = file.read().strip()
            self.vlm = AzureOpenAI(api_key=api_key,
                    api_version= '2024-04-01-preview',
                    azure_endpoint= 'https://gpt-4o-pjm.openai.azure.com/'
                    )
            self.embedding = AzureOpenAI(api_key=api_key_e,
                    api_version= "2024-02-15-preview",
                    azure_endpoint= "https://text-embedding-3-large-pjm.openai.azure.com/"
                    )
        else:
            with open(os.path.join(ROOT_DIR, 'api_key/api_key.txt'), 'r', encoding='utf-8') as file:
                api_key = file.read().strip()
            self.embedding = self.vlm = OpenAI(api_key=api_key)
        self.verbose = verbose
        self.deployment_name = 'gpt-4o'
        self.deployment_name_e = "text-embedding-3-large"
        self.token_calculator = TokenCalculate('gpt-4o')
        self.try_times = try_times

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        self.token_calculator.accumulate_token(embedding=text)
        return self.embedding.embeddings.create(input = [text], model=self.deployment_name_e).data[0].embedding
    
    def get_answer(self, template_type, **kwargs):
        image_buf = kwargs.get('image', None)
        cnt = 0
        prompt = generate_prompt(template_type, **kwargs)
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        if image_buf:
            img_encoded = base64.b64encode(image_buf.getvalue()).decode('utf-8')
            image_buf.close()
            item =  {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_encoded}",
                        "detail": "high"
                    }
                }
            content.append(item)
            self.token_calculator.accumulate_token(image_num=1)

        while cnt < self.try_times:
            try:
                response = self.vlm.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {
                            "role": "user",
                            "content": content
                            },
                        ],
                        max_tokens=2048,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                )
                result = response.choices[0].message.content
                self.token_calculator.accumulate_token(prompt=prompt, result=result)
                break
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        if self.verbose:
            print('prompt:', prompt)
            print('result:', result)
        return result



######################## prompt ########################
class TokenCalculate:
    def __init__(self, model_type):
        self.token_calculate = tiktoken.encoding_for_model(model_type)
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
        self.images = 0

    def accumulate_token(self, prompt=None, result=None, embedding = None, image_num = None):
        if prompt:
            self.input_tokens += len(self.token_calculate.encode(prompt))
        if result:
            self.output_tokens += len(self.token_calculate.encode(result))
        if embedding:
            self.embedding_tokens += len(self.token_calculate.encode(embedding))
        if image_num:
            self.images += image_num

    def calculate_money(self, money_per_image, money_per_input_token, money_per_output_token, money_per_embedding_token):
        money = money_per_image*self.images + money_per_input_token*self.input_tokens + money_per_output_token*self.output_tokens + money_per_embedding_token*self.embedding_tokens
        print("USD: ", money)
        return money

def generate_prompt(template_type, **kwargs):
    """
    Generate a complete prompt based on the template type and provided content.

    Parameters:
        template_type (str): The type of template to use.
        **kwargs: The content to fill in the template.

    Returns:
        str: The complete prompt.
    """

    if template_type == 'get_goal':
        template = (
                "To better answer the question, the agent first should understand"
                    +" which object is the core in the question."
                    +"\nPlease give the output without any adjective according to the examples below:"
                    +"\ninput: Can you locate the cabinet that has a single faucet hole without an installed faucet?"
                    +"\noutput: cabinet"
                    +"\ninput: What's the color of the blanket in the living room?"
                    +"\noutput: blanket"
                    +"\ninput: {question}"
                    +"\noutput:"
            )
    elif template_type == 'extract_info':
        template = (
            "Given the dialogue which always has a question and an answer, "
            +"please summarize the information you get from the input question and answer in a short and precise sentence."
            +"\nYou can give the output according to the examples below:"
            +"\nPlease give the output without any adjective according to the examples below:"
            +"\ninput_question: Is this the cabinet you are looking for? input_answer: No, that's not the cabinet I'm looking for."
            +"\noutput: The goal cabinet hasn't been found yet."
            +"\ninput_question: Could you please describe the location of the cabinet in relation to other objects or landmarks in the room? input_answer: There are no bottles on the cabinet."
            +"\noutput: The goal cabinet has no bottles on it."
            +"\ninput question:{question} input answer:{answer}"
            +"\noutput:"
        )
    elif template_type == 'get_info_type':
        template = (
            "Which type do you think the information below belongs to, location, spatial relation with other objects, appearance description or others?"
            +"\n{info}"
            +"\nPlease choose your answer from one of the four words: location, spatial, appearance, others."
            +"\nAnswer:"
        )
    # elif template_type == 'score_candidate':
    #     template = (
    #         "Given a candidate with the following features:\n"
    #         + "{description}\n"
    #         + "\nAnd the goal {goal}'s information characterized by:\n"
    #         + "{goal_info}\n"
    #         + "\nPlease assess the probability that the candidate is the goal. "
    #         + "Rate the probability on a scale from 0 to 10, where 0 indicates the candidate can not be the goal. "
    #         + "and 10 indicates the candidate must be the goal. Provide a single integer as the score."
    #         + "\n\nRating: "
    #     )
    elif template_type == 'choose_candidate':
        template = (
            "USER:Here are the descriptions of the current candidates for the goal object {goal}:\n"
            + "{description}\n"
            + "Here are the known information about the goal object {goal}:\n"
            + "{goal_info}\n"
            + "Each line of candidate description corresponds to a candidate." 
            + "The number in the description is the candidate's index,"
            + "and the text after \':\' is the candidate's description."
            + "\nNow, based on the provided information about the goal object,"
            + "please select the candidate most likely to be the goal object."
            + "\nYou only need to output the candidate's index."
            + "Please do not output anything other than the candidate's index."
            + "\nASSISTANT:")
    elif template_type == 'ask_question':
        template = (
            "USER:Here are the descriptions of the current candidates for the goal object {goal}:\n"
            + "{description}\n"
            + "Here are the known information about the goal object {goal}:\n"
            + "{goal_info}\n"
            + "Now you can ask a question about the goal object. "
            + "Based on the information described above, "
            + "what question do you think will help to minimize the scope of the possible candidates? "
            + "Just output the question, don't include the reason or explanation."
            + "\nExample 1: Could you please give me more information about the goal object?"
            + "\nExample 2: Where is the goal object? (This question will help a lot if you don't know where is the goal object and you know the location of most of the candidates.)"
            + "\nASSISTANT:")
    # elif template_type == 'format_description':
    #     template = (
    #             "Please extract and rearrange the information related to goal {goal} in Input as order like this:"
    #             +"\nroom information if has; relation information if has; appearance information if has."
    #             +"\n\nHere are some examples:"
    #             +"\nInput: The chair is not located in the bedroom."
    #             +"\nOutput:"
    #             +"\nThe goal chair is not in the bedroom." 
    #             +"\n\nInput: It has a rectangular shape with rounded edges. There is a vase on the cabinet.(The goal object is cabinet)"
    #             +"\nOutput:"
    #             +"\nThe goal cabinet has a vase on it. The goal cabinet has a rectangular shape with rounded edges."
    #             +"\n\nInput: {description}"
    #             +"\n\nOutput:" 
    #             )
    elif template_type == 'get_description':
        # template = ("USER: <image>\nAnswer the folllwing questions:"
        #               +"\nWhere is the {goal} in region {region}?"
        #               +"\nWhat is the appearance of the {goal} in region {region}?"
        #               +"\nWhat is the spatial relation between the {goal} in region {region} and other objects nearby?"
        #               +"\nIf there isn't a {goal} in region {region}, just output -1."
        #               +"\nASSISTANT:")
        template = ("USER: Please tell me is there any {goal} in the given image. "
                    + "If no, just output -1. If yes, please firstly give the region number that "
                    + "the {goal} is/are located in and secondly answer the following questions about the {goal} if you can:"
                    + "\nWhere is the {goal} located in?(e.g. kitchen, bedroom, bathroom, etc.)"
                    + "\nWhat is the appearance of the {goal}?"
                    + "\nWhat is the spatial relation between the {goal} and other objects nearby?"
                    + "\nPlease be concise and you don't need to explain the reasons of the answers," 
                    + "and connect the answers of the questions by \';\'."
                    + "\nEvery {goal} in the image should have a relative region number and the corresponding answers connected by \';\'."
                    + "\nPlease give each pair of region number and answers in one line."
                    + "\nRemember separate the region number and answers by \':\'. "
                    + "(Output format: Region [region number]:answers)"
                    + "\nASSISTANT:")
    elif template_type == 'make_decision':
        template = vlm_prompt
    else:
        raise ValueError(f"Template type '{template_type}' not found.")
    prompt = template.format(**kwargs)
    return prompt
