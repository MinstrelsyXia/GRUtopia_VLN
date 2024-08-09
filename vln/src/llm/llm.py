import os,sys
import base64
import requests
import yaml
from openai import AzureOpenAI, OpenAI

from vln.src.utils.utils import dict_to_namespace

from grutopia.core.util.log import log

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OpenAI_GPT:
    def __init__(self, args, verbose = False, try_times=20):
        self.args = args
        self.llm_args = args.llms
        if self.llm_args.azure:
            # copy from llm_agent/utils/large_models.py
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

        self.verbose = verbose
        self.try_times = try_times
        self.token_manager = TokenCalculate(self.llm_args.model_name)

        # init prompt
        self.system_prompt = self.init_system_prompt(self.llm_args.system_prompt_file)
        self.history = []
    
    def encode_image(self, image_path):
        # Function to encode the image
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_answer(self, template_type, **kwargs):
        cnt = 0
        # Process the text
        prompt = generate_prompt(template_type, **kwargs)
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        # Process the image
        images = kwargs.get('images', None)
        if images is not None:
            for image_path in images:
                base64_image = self.encode_image(image_path)
                item =  {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": self.llm_args.image_detail
                        }
                    }
                content.append(item)

        while cnt < self.try_times:
            try:
                result, usage_info = self.get_response(content)
                self.token_manager.update_token_nums(usage_info)
                break
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        if self.verbose:
            print('prompt:', prompt)
            print('result:', result)
        return result

    def init_system_prompt(self, system_prompt_file):
        with open(system_prompt_file, 'r', encoding='utf-8') as file:
            sys_content = file.read()
        system_msg = {
            "role": "developer",
            "content": sys_content
        }
        if self.verbose:
            log.info(f"System_msg has initialized as:\n{self.system_prompt}")
        return system_msg

    def post_requests(self, msg):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_args.api_key}"
            }
        payload = {
            "model": f"{self.llm_args.model_name}",
            "messages": msg,
            "max_tokens": self.llm_args.max_tokens,
            "interactive": self.llm_args.interactive 
            }
        response = requests.post(self.llm_args.api_url, headers=headers, json=payload)
        response_msg = response["choices"][0]["message"]["content"]
        usage_info = response["usage"]
        return response_msg, usage_info

    def generate_prompt(self, sys_prompt=None, usr_prompt=None, img_list=[], **kwargs):
        content = []
        if sys_prompt is not None:
            if isinstance(sys_prompt, dict):
                

        


######################## prompt ########################
class TokenCalculate:
    def __init__(self, model_type):
        self.model_type = model_type
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def update_token_nums(self, usage_info):
        self.prompt_tokens += usage_info['prompt_tokens']
        self.completion_tokens += usage_info['completion_tokens']
        self.total_tokens += usage_info['total_tokens']

    def cost_money(self):
        # TODO
        return 


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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test LLM for VLN")
    parser.add_argument("--vln_cfg_file", type=str, default="vln/configs/vln_cfg.yaml")
    args = parser.parse_args()
    '''Init VLN config'''
    with open(args.vln_cfg_file, 'r') as f:
        vln_config = dict_to_namespace(yaml.load(f.read(), yaml.FullLoader))
    # update args into vln_config
    for key, value in vars(args).items():
        setattr(vln_config, key, value)

    vlm_agent = OpenAI_GPT(vln_config)
    # read system prompt
    system_prompt_file = os.path.join(ROOT_DIR, 'prompts', 'system_info.txt')
    with open(system_prompt_file, 'r', encoding='utf-8') as file:
        system_message = file.read()
    

    # test_prompt = '''I need you to guide a robot to follow the given instructions to navigate indoor. I would provide you an image with full semantic information, and a top-down map with the robot's current position, and its ego-centric observation. You need to first describe the robot's current state, and inference the most possible next waypoints based on the given information. You can output the possible coordinates, or the possible region on the map. If you think the robot should turn around to get more information, you can also output the order to let robot explore more.
    # The given instruction is: Exit the bedroom, cross the room to the left and stop near the room divider on the left.
    # The given visual semantic image is: '''
    image_path = "/home/wangliuyi/code/w61-grutopia/vln/src/llm/test_imgs/5593/BuildMap_20240808v1.png"
    images = [image_path]
    results = vlm_agent.get_answer_test(test_prompt, images)
    print(results)