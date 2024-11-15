import os
import openai


def parse_object_goal_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    question = f"""
    I: go to the kitchen and then go to the toilet. A: kitchen, toilet
    I: go to the chair and then go to another chair. A: chair, chair
    I: navigate to the green sofa and turn right and find several chairs, finally go to the painting. A: green sofa, chairs, painting
    I: approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen. A: window, television, oven, kitchen
    I: walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed. A: plant, table, bedroom, bed
    I: go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room. A: stairs, room, book shelf, table, next room
    I: Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances. A: table, cushion, column, appliances.
    I: Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table. A: chair, table
    I: {language_instr}. A:"""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop=None,
    )
    result = response["choices"][0]["text"].strip()
    print("landmarks: ", result)
    return [x.strip() for x in result.split(",")]

# def parse_object_goal_instruction(language_instr):
#     """
#     Parse language instruction into a series of landmarks
#     Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
#     """
#     import openai

#     openai_key = os.environ["OPENAI_KEY"]
#     openai.api_key = openai_key
#     client = openai.OpenAI(api_key=openai_key)
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": "go to the kitchen and then go to the toilet",
#             },
#             {
#                 "role": "assistant",
#                 "content": "kitchen, toilet"
#             },
#             {
#                 "role": "user",
#                 "content": "go to the chair and then go to another chair"
#             },
#             {
#                 "role": "assistant",
#                 "content": "chair, chair"
#             },
#             {
#                 "role": "user",
#                 "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting"
#             },
#             {
#                 "role": "assistant",
#                 "content": "green sofa, chairs, painting"
#             },
#             {
#                 "role": "user",
#                 "content": "approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen"
#             },
#             {
#                 "role": "assistant",
#                 "content": "window, television, oven, kitchen"
#             },
#             {
#                 "role": "user",
#                 "content": "walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed"
#             },
#             {
#                 "role": "assistant",
#                 "content": "plant, table, bedroom, bed"
#             },
#             {
#                 "role": "user",
#                 "content": "go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room"
#             },
#             {
#                 "role": "assistant",
#                 "content": "stairs, room, book shelf, table, next room"
#             },
#             {
#                 "role": "user",
#                 "content": "Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances"
#             },
#             {
#                 "role": "assistant",
#                 "content": "table, cushion, column, appliances"
#             },
#             {
#                 "role": "user",
#                 "content": "Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table"
#             },
#             {
#                 "role": "assistant",
#                 "content": "chair, table"
#             },
#             {
#                 "role": "user",
#                 "content": language_instr
#             }
#         ],
#         max_tokens=300,
#     )

#     text = response.choices[0].message.content
#     return [x.strip() for x in text.split(",")]


def parse_object_goal_instruction(language_instr):
    """
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    from openai import AzureOpenAI
    try:
        with open('api_key/azure_api_key.txt', 'r', encoding='utf-8') as file:
            api_key = file.read().strip()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    client =AzureOpenAI(
                api_key=api_key,
                api_version='2024-02-15-preview',
                azure_endpoint='https://gpt-4o-pjm.openai.azure.com/'
            )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "go to the kitchen and then go to the toilet",
            },
            {
                "role": "assistant",
                "content": "kitchen, toilet"
            },
            {
                "role": "user",
                "content": "go to the chair and then go to another chair"
            },
            {
                "role": "assistant",
                "content": "chair, chair"
            },
            {
                "role": "user",
                "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting"
            },
            {
                "role": "assistant",
                "content": "green sofa, chairs, painting"
            },
            {
                "role": "user",
                "content": "approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen"
            },
            {
                "role": "assistant",
                "content": "window, television, oven, kitchen"
            },
            {
                "role": "user",
                "content": "walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed"
            },
            {
                "role": "assistant",
                "content": "plant, table, bedroom, bed"
            },
            {
                "role": "user",
                "content": "go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room"
            },
            {
                "role": "assistant",
                "content": "stairs, room, book shelf, table, next room"
            },
            {
                "role": "user",
                "content": "Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances"
            },
            {
                "role": "assistant",
                "content": "table, cushion, column, appliances"
            },
            {
                "role": "user",
                "content": "Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table"
            },
            {
                "role": "assistant",
                "content": "chair, table"
            },
            {
                "role": "user",
                "content": language_instr
            }
        ],
        max_tokens=300,
    )

    text = response.choices[0].message.content
    return [x.strip() for x in text.split(",")]




def parse_spatial_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    """
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        question = f"""
# move a bit to the right of the refrigerator.
robot.move_to_right('refrigerator')
###
# move in between the couch and bookshelf.
robot.move_in_between('couch', 'bookshelf')
###
# move in the middle of the stairs and oven.
robot.move_in_between('stairs', 'oven')
###
# face the toilet.
robot.face('toilet')
###
# move to the south side of the table
robot.move_south('table')
###
# move to the west of the chair
robot.move_west('chair')
###
# move to the east of the chair
robot.move_east('chair')
###
# turn left 60 degrees
robot.turn(-60)
###
# turn right 20 degrees
robot.turn(20)
###
# find any chairs in the environment
robot.move_to_object('chair')
###
# come to the table
robot.move_to_object('table')
###
# with the television on your left
robot.with_object_on_left('television')
###
# with the toilet to your right, turn left 30 degrees
robot.with_object_on_right('toilet')
robot.turn(-30)
###
# with the toilet to your left, turn right 30 degrees
robot.with_object_on_left('toilet')
robot.turn(30)
###
# with the television behind you
robot.face('television')
robot.turn(180)
###
# go to the south of the table in front of you, with the table on your left, turn right 45 degrees
robot.move_south('table')
robot.with_object_on_left('table')
robot.turn(45)
###
# move forward for 3 meters
robot.move_forward(3)
###
# move right 2 meters
robot.turn(90)
robot.move_forward(2)
###
# move left 3 meters
robot.turn(-90)
robot.move_forward(3)
###
# move back and forth between the chair and the table 3 times
pos1 = robot.get_pos('chair')
pos2 = robot.get_pos('table')
for i in range(3):
    robot.move_to(pos1)
    robot.move_to(pos2)
###
# go to the television
robot.move_to_object('television')
###
# monitor the chair from 4 view points
contour = robot.get_contour('chair')
np.random.shuffle(contour)
for pos in contour[:4]:
    robot.move_to(pos)
    robot.face('chair')
###
# navigate to 3 meters right of the table
robot.move_to_right('table')
robot.face('table')
robot.turn(180)
robot.move_forward(3)
###
# turn west
robot.turn_absolute(-90)
###
# turn east
robot.turn_absolute(90)
###
# turn south
robot.turn_absolute(180)
###
# turn north
robot.turn_absolute(0)
###
# turn east and then turn left 90 degrees
robot.turn_absolute(90)
robot.turn(-90)
###
# move 3 meters north of the chair
robot.move_north('chair')
robot.turn_absolute(0)
robot.move_forward(3)
###
# move 1 meters south of the chair
robot.move_south('chair')
robot.turn_absolute(180)
robot.move_forward(3)
###
# move 3 meters west
robot.turn_absolute(-90)
robot.move_forward(3)
# {lang}
    """
        print("lang: ", lang)
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=question,
            max_tokens=400,
            temperature=0.0,
            stop="###",
        )
        result = response["choices"][0]["text"].strip()
        if result:
            results += result + "\n"
        print(result)
    print("generated code")
    print(results)
    return results

def parse_spatial_instruction(language_instr):
    import openai

    from openai import AzureOpenAI
    try:
        with open('api_key/azure_api_key.txt', 'r', encoding='utf-8') as file:
            api_key = file.read().strip()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        client =AzureOpenAI(
                    api_key=api_key,
                    api_version='2024-02-15-preview',
                    azure_endpoint='https://gpt-4o-pjm.openai.azure.com/'
                )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "move a bit to the right of the refrigerator"},
                {"role": "assistant", "content": "self.move_to_right('refrigerator')"},
                {"role": "user", "content": "move in between the couch and bookshelf"},
                {"role": "assistant", "content": "self.move_in_between('couch', 'bookshelf')"},
                {"role": "user", "content": "move in the middle of the stairs and oven"},
                {"role": "assistant", "content": "self.move_in_between('stairs', 'oven')"},
                {"role": "user", "content": "face the toilet"},
                {"role": "assistant", "content": "self.face('toilet')"},
                {"role": "user", "content": "move to the south side of the table"},
                {"role": "assistant", "content": "self.move_south('table')"},
                {"role": "user", "content": "move to the west of the chair"},
                {"role": "assistant", "content": "self.move_west('chair')"},
                {"role": "user", "content": "move to the east of the chair"},
                {"role": "assistant", "content": "self.move_east('chair')"},
                {"role": "user", "content": "turn left 60 degrees"},
                {"role": "assistant", "content": "self.turn(-60)"},
                {"role": "user", "content": "turn right 20 degrees"},
                {"role": "assistant", "content": "self.turn(20)"},
                {"role": "user", "content": "find any chairs in the environment"},
                {"role": "assistant", "content": "self.move_to_object('chair')"},
                {"role": "user", "content": "come to the table"},
                {"role": "assistant", "content": "self.move_to_object('table')"},
                {"role": "user", "content": "with the television on your left"},
                {"role": "assistant", "content": "self.with_object_on_left('television')"},
                {"role": "user", "content": "with the toilet to your right, turn left 30 degrees"},
                {"role": "assistant", "content": "self.with_object_on_right('toilet')\nself.turn(-30)"},
                {"role": "user", "content": "with the toilet to your left, turn right 30 degrees"},
                {"role": "assistant", "content": "self.with_object_on_left('toilet')\nself.turn(30)"},
                {"role": "user", "content": "with the television behind you"},
                {"role": "assistant", "content": "self.face('television')\nself.turn(180)"},
                {"role": "user", "content": "go to the south of the table in front of you, with the table on your left, turn right 45 degrees"},
                {"role": "assistant", "content": "self.move_south('table')\nself.with_object_on_left('table')\nself.turn(45)"},
                {"role": "user", "content": "move forward for 3 meters"},
                {"role": "assistant", "content": "self.move_forward(3)"},
                {"role": "user", "content": "move right 2 meters"},
                {"role": "assistant", "content": "self.turn(90)\nself.move_forward(2)"},
                {"role": "user", "content": "move left 3 meters"},
                {"role": "assistant", "content": "self.turn(-90)\nself.move_forward(3)"},
                {"role": "user", "content": "move back and forth between the chair and the table 3 times"},
                {"role": "assistant", "content": "pos1 = self.get_pos('chair')\npos2 = self.get_pos('table')\nfor i in range(3):\n    self.move_to(pos1)\n    self.move_to(pos2)"},
                {"role": "user", "content": "go to the television"},
                {"role": "assistant", "content": "self.move_to_object('television')"},
                {"role": "user", "content": "monitor the chair from 4 view points"},
                {"role": "assistant", "content": "contour = self.get_contour('chair')\nnp.random.shuffle(contour)\nfor pos in contour[:4]:\n    self.move_to(pos)\n    self.face('chair')"},
                {"role": "user", "content": "navigate to 3 meters right of the table"},
                {"role": "assistant", "content": "self.move_to_right('table')\nself.face('table')\nself.turn(180)\nself.move_forward(3)"},
                {"role": "user", "content": "turn west"},
                {"role": "assistant", "content": "self.turn_absolute(-90)"},
                {"role": "user", "content": "turn east"},
                {"role": "assistant", "content": "self.turn_absolute(90)"},
                {"role": "user", "content": "turn south"},
                {"role": "assistant", "content": "self.turn_absolute(180)"},
                {"role": "user", "content": "turn north"},
                {"role": "assistant", "content": "self.turn_absolute(0)"},
                {"role": "user", "content": "turn east and then turn left 90 degrees"},
                {"role": "assistant", "content": "self.turn_absolute(90)\nself.turn(-90)"},
                {"role": "user", "content": "move 3 meters north of the chair"},
                {"role": "assistant", "content": "self.move_north('chair')\nself.move_forward(3)"},
                {"role": "user", "content": "move 1 meters south of the chair"},
                {"role": "assistant", "content": "self.move_south('chair')\nself.move_forward(3)"},
                {"role": "user", "content": "move 3 meters west"},
                {"role": "assistant", "content": "self.turn_absolute(-90)\nself.move_forward(3)"},
                {"role": "user", "content": "Go around the right side of the center unit and stop by the right side doorway with the dining table and mirror in it"},
                {"role": "assistant", "content": "self.move_to_right('center unit')\nself.move_to_right('doorway')\nself.face('dining table')\nself.face('mirror')"},
                {"role": "user", "content": "Go around the kitchen island and wait between the tall cabinet and wine fridge."},
                {"role": "assistant", "content": "self.move_to('kitchen island')\nself.move_in_between('cabinet','wine fridge')"},
                {"role": "user", "content": "Turn to the right, go past the refrigerator.  Turn left and walk to the point where you're to the hallway by the entry and dining room area."},
                {"role": "assistant", "content": "self.turn(90)\nself.move_to_object('refrigerator')\nself.turn(-90)\nself.move_to('hallway')\nself.move_in_between('entry','dining roon area')"},
                {"role": "user", "content": "Leave the massage room and take a right. Go into the first door on the right with a tanning bed. Stop and the foot of the tanning bed "},
                {"role": "assistant", "content": "self.move_to('door')\nself.move_forward(1)\nself.turn(90)\nself.move_in_between('door','tanning bed')\nself.face('tanning bed')"},
                {"role": "user", "content": "Exit the bathroom, head straight, turn right at the wall, then turn right again and stop in the doorway by the table. "},
                {"role": "assistant", "content": "self.move_to('bathroom')\nself.move_to('door')\nself.move_forward(1)\nself.turn(90)\nself.move_to('table')"},                                   
                {"role": "user", "content": f"{lang}. Please only answer the functions. Functions to use are: self.move_to_left(object_name), self.move_to_right(object_name), self.get_pos(object_name), self.get_contour(object_name), self.with_object_on_left(object_name), self.with_object_on_right(obect_name) ,self.move_in_between(object_a, object_b), self.face(object_name), self.turn_absolute(angle),  self.move_forward(distance), self.turn(angle), self.move_to_object(object_name)"},
            ],
            # self.move_north(object_name), self.move_south(object_name), self.move_east(object_name), self.move_west(object_name),
            max_tokens=150,
        )
        text = response.choices[0].message.content
        if text:
            results += text + "\n"
    return text


# def parse_spatial_instruction(language_instr):
#     import openai

#     openai_key = os.environ["OPENAI_KEY"]
#     openai.api_key = openai_key
#     # instructions_list = language_instr.split(",")
#     instructions_list = [language_instr]
#     results = ""
#     for lang in instructions_list:
#         client = openai.OpenAI(api_key=openai_key)
#         response = client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "user", "content": "move a bit to the right of the refrigerator"},
#                 {"role": "assistant", "content": "robot.move_to_right('refrigerator')"},
#                 {"role": "user", "content": "move in between the couch and bookshelf"},
#                 {"role": "assistant", "content": "robot.move_in_between('couch', 'bookshelf')"},
#                 {"role": "user", "content": "move in the middle of the stairs and oven"},
#                 {"role": "assistant", "content": "robot.move_in_between('stairs', 'oven')"},
#                 {"role": "user", "content": "face the toilet"},
#                 {"role": "assistant", "content": "robot.face('toilet')"},
#                 {"role": "user", "content": "move to the south side of the table"},
#                 {"role": "assistant", "content": "robot.move_south('table')"},
#                 {"role": "user", "content": "move to the west of the chair"},
#                 {"role": "assistant", "content": "robot.move_west('chair')"},
#                 {"role": "user", "content": "move to the east of the chair"},
#                 {"role": "assistant", "content": "robot.move_east('chair')"},
#                 {"role": "user", "content": "turn left 60 degrees"},
#                 {"role": "assistant", "content": "robot.turn(-60)"},
#                 {"role": "user", "content": "turn right 20 degrees"},
#                 {"role": "assistant", "content": "robot.turn(20)"},
#                 {"role": "user", "content": "find any chairs in the environment"},
#                 {"role": "assistant", "content": "robot.move_to_object('chair')"},
#                 {"role": "user", "content": "come to the table"},
#                 {"role": "assistant", "content": "robot.move_to_object('table')"},
#                 {"role": "user", "content": "with the television on your left"},
#                 {"role": "assistant", "content": "robot.with_object_on_left('television')"},
#                 {"role": "user", "content": "with the toilet to your right, turn left 30 degrees"},
#                 {"role": "assistant", "content": "robot.with_object_on_right('toilet')\nrobot.turn(-30)"},
#                 {"role": "user", "content": "with the toilet to your left, turn right 30 degrees"},
#                 {"role": "assistant", "content": "robot.with_object_on_left('toilet')\nrobot.turn(30)"},
#                 {"role": "user", "content": "with the television behind you"},
#                 {"role": "assistant", "content": "robot.face('television')\nrobot.turn(180)"},
#                 {"role": "user", "content": "go to the south of the table in front of you, with the table on your left, turn right 45 degrees"},
#                 {"role": "assistant", "content": "robot.move_south('table')\nrobot.with_object_on_left('table')\nrobot.turn(45)"},
#                 {"role": "user", "content": "move forward for 3 meters"},
#                 {"role": "assistant", "content": "robot.move_forward(3)"},
#                 {"role": "user", "content": "move right 2 meters"},
#                 {"role": "assistant", "content": "robot.turn(90)\nrobot.move_forward(2)"},
#                 {"role": "user", "content": "move left 3 meters"},
#                 {"role": "assistant", "content": "robot.turn(-90)\nrobot.move_forward(3)"},
#                 {"role": "user", "content": "move back and forth between the chair and the table 3 times"},
#                 {"role": "assistant", "content": "pos1 = robot.get_pos('chair')\npos2 = robot.get_pos('table')\nfor i in range(3):\n    robot.move_to(pos1)\n    robot.move_to(pos2)"},
#                 {"role": "user", "content": "go to the television"},
#                 {"role": "assistant", "content": "robot.move_to_object('television')"},
#                 {"role": "user", "content": "monitor the chair from 4 view points"},
#                 {"role": "assistant", "content": "contour = robot.get_contour('chair')\nnp.random.shuffle(contour)\nfor pos in contour[:4]:\n    robot.move_to(pos)\n    robot.face('chair')"},
#                 {"role": "user", "content": "navigate to 3 meters right of the table"},
#                 {"role": "assistant", "content": "robot.move_to_right('table')\nrobot.face('table')\nrobot.turn(180)\nrobot.move_forward(3)"},
#                 {"role": "user", "content": "turn west"},
#                 {"role": "assistant", "content": "robot.turn_absolute(-90)"},
#                 {"role": "user", "content": "turn east"},
#                 {"role": "assistant", "content": "robot.turn_absolute(90)"},
#                 {"role": "user", "content": "turn south"},
#                 {"role": "assistant", "content": "robot.turn_absolute(180)"},
#                 {"role": "user", "content": "turn north"},
#                 {"role": "assistant", "content": "robot.turn_absolute(0)"},
#                 {"role": "user", "content": "turn east and then turn left 90 degrees"},
#                 {"role": "assistant", "content": "robot.turn_absolute(90)\nrobot.turn(-90)"},
#                 {"role": "user", "content": "move 3 meters north of the chair"},
#                 {"role": "assistant", "content": "robot.move_north('chair')\nrobot.move_forward(3)"},
#                 {"role": "user", "content": "move 1 meters south of the chair"},
#                 {"role": "assistant", "content": "robot.move_south('chair')\nrobot.move_forward(3)"},
#                 {"role": "user", "content": "move 3 meters west"},
#                 {"role": "assistant", "content": "robot.turn_absolute(-90)\nrobot.move_forward(3)"},
#                 {"role": "user", "content": lang},
#             ],
#             max_tokens=300,
#         )
#         text = response.choices[0].message.content
#         if text:
#             results += text + "\n"
#     return text

def generate_prompt(template_type, **kwargs):
    """
    根据模板类型和传入的参数生成提示语（prompt）。
    
    :param template_type: 字符串，指定模板类型。
    :param kwargs: 额外的参数字典，用于生成提示内容。
    :return: 根据模板生成的提示语（prompt）。
    """
    prompt = ""

    if template_type == "parse_spatial_instruction":
        # 针对聊天类型的提示
        user_message = kwargs.get('user_message', '')
        prompt = f"User: {user_message}\nAssistant:"

    elif template_type == "qa":
        # 针对问答类型的提示
        question = kwargs.get('question', '')
        context = kwargs.get('context', '')
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    elif template_type == "summary":
        # 针对摘要类型的提示
        text_to_summarize = kwargs.get('text', '')
        prompt = f"Please summarize the following text:\n{text_to_summarize}"

    elif template_type == "classification":
        # 针对分类任务的提示
        text_to_classify = kwargs.get('text', '')
        prompt = f"Classify the following text into appropriate categories:\n{text_to_classify}"

    elif template_type == "code_generation":
        # 针对代码生成的提示
        task_description = kwargs.get('task_description', '')
        prompt = f"Write a Python function that accomplishes the following task:\n{task_description}"

    elif template_type == "custom":
        # 自定义类型的提示，用户可以提供额外的自定义参数
        custom_prompt = kwargs.get('custom_prompt', '')
        prompt = custom_prompt
    elif template_type == "test":
        # 测试类型的提示
        prompt = "Hello, World!"
    else:
        # 默认提示，若没有匹配的模板类型
        prompt = "Please provide a valid template_type."

    return prompt


from openai import AzureOpenAI
from openai import OpenAI
import base64
class Vlm_gpt4o:
    def __init__(self, azure=False, verbose=False, try_times=20):
        self.verbose = verbose
        self.try_times = try_times

        if azure:
            # 读取Azure的API密钥
            self.api_key, self.api_key_e = self._read_api_keys(
                'api_key/azure_api_key.txt',
                'api_key/azure_api_key_e.txt'
            )
            # 初始化Azure OpenAI实例
            self.vlm = AzureOpenAI(
                api_key=self.api_key,
                api_version='2024-04-01-preview',
                azure_endpoint='https://gpt-4o-pjm.openai.azure.com/'
            )
            # self.embedding = AzureOpenAI(
            #     api_key=self.api_key_e,
            #     api_version='2024-02-15-preview',
            #     azure_endpoint='https://text-embedding-3-large-pjm.openai.azure.com/'
            # )
        else:
            # 读取普通OpenAI的API密钥
            self.api_key = self._read_api_key('api_key/api_key.txt')
            # 初始化OpenAI实例
            self.vlm = OpenAI(api_key=self.api_key)
            self.embedding = self.vlm

        self.deployment_name = 'gpt-4o'
        # self.deployment_name_e = 'text-embedding-3-large'
        # self.token_calculator = TokenCalculate('gpt-4o')

    def _read_api_keys(self, api_key_path, api_key_e_path):
        """读取API密钥并返回"""
        try:
            with open(api_key_path, 'r', encoding='utf-8') as file:
                api_key = file.read().strip()
            with open(api_key_e_path, 'r', encoding='utf-8') as file:
                api_key_e = file.read().strip()
            return api_key, api_key_e
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    def _read_api_key(self, api_key_path):
        """读取单个API密钥"""
        try:
            with open(api_key_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    def get_embedding(self, text):
        """获取文本的嵌入向量"""
        text = text.replace("\n", " ")
        self.token_calculator.accumulate_token(embedding=text)
        try:
            response = self.embedding.embeddings.create(
                input=[text], 
                model=self.deployment_name_e
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Failed to get embedding: {e}")
            return None

    def get_answer(self, template_type, **kwargs):
        """生成GPT-4O的答案"""
        image_buf = kwargs.get('image', None)
        cnt = 0
        result = None
        prompt = generate_prompt(template_type, **kwargs)

        # 生成请求内容
        content = [{"type": "text", "text": prompt}]
        if image_buf:
            img_encoded = base64.b64encode(image_buf.getvalue()).decode('utf-8')
            image_buf.close()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_encoded}",
                    "detail": "high"
                }
            })
            self.token_calculator.accumulate_token(image_num=1)

        # 尝试多次请求以应对异常情况
        while cnt < self.try_times:
            try:
                response = self.vlm.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                result = response.choices[0].message.content
                self.token_calculator.accumulate_token(prompt=prompt, result=result)
                break
            except Exception as e:
                print(f"Attempt {cnt+1} failed: {e}")
                cnt += 1

        # 日志输出
        if self.verbose:
            print(f"Prompt: {prompt}")
            print(f"Result: {result}")

        return result

    def parse_spatial_instruction(self, language_instr):
        """解析空间指令"""
        instructions_list = [language_instr]
        results = ""
        
        for lang in instructions_list:
            try:
                response = self.vlm.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "user", "content": "move a bit to the right of the refrigerator"},
                        {"role": "assistant", "content": "robot.move_to_right('refrigerator')"},
                        {"role": "user", "content": "move in between the couch and bookshelf"},
                        {"role": "assistant", "content": "robot.move_in_between('couch', 'bookshelf')"},
                        {"role": "user", "content": "move in the middle of the stairs and oven"},
                        {"role": "assistant", "content": "robot.move_in_between('stairs', 'oven')"},
                        {"role": "user", "content": "face the toilet"},
                        {"role": "assistant", "content": "robot.face('toilet')"},
                        {"role": "user", "content": "move to the south side of the table"},
                        {"role": "assistant", "content": "robot.move_south('table')"},
                        {"role": "user", "content": "move to the west of the chair"},
                        {"role": "assistant", "content": "robot.move_west('chair')"},
                        {"role": "user", "content": "move to the east of the chair"},
                        {"role": "assistant", "content": "robot.move_east('chair')"},
                        {"role": "user", "content": "turn left 60 degrees"},
                        {"role": "assistant", "content": "robot.turn(-60)"},
                        {"role": "user", "content": "turn right 20 degrees"},
                        {"role": "assistant", "content": "robot.turn(20)"},
                        {"role": "user", "content": "find any chairs in the environment"},
                        {"role": "assistant", "content": "robot.move_to_object('chair')"},
                        {"role": "user", "content": "come to the table"},
                        {"role": "assistant", "content": "robot.move_to_object('table')"},
                        {"role": "user", "content": "with the television on your left"},
                        {"role": "assistant", "content": "robot.with_object_on_left('television')"},
                        {"role": "user", "content": "with the toilet to your right, turn left 30 degrees"},
                        {"role": "assistant", "content": "robot.with_object_on_right('toilet')\nrobot.turn(-30)"},
                        {"role": "user", "content": "with the toilet to your left, turn right 30 degrees"},
                        {"role": "assistant", "content": "robot.with_object_on_left('toilet')\nrobot.turn(30)"},
                        {"role": "user", "content": "with the television behind you"},
                        {"role": "assistant", "content": "robot.face('television')\nrobot.turn(180)"},
                        {"role": "user", "content": "go to the south of the table in front of you, with the table on your left, turn right 45 degrees"},
                        {"role": "assistant", "content": "robot.move_south('table')\nrobot.with_object_on_left('table')\nrobot.turn(45)"},
                        {"role": "user", "content": "move forward for 3 meters"},
                        {"role": "assistant", "content": "robot.move_forward(3)"},
                        {"role": "user", "content": "move right 2 meters"},
                        {"role": "assistant", "content": "robot.turn(90)\nrobot.move_forward(2)"},
                        {"role": "user", "content": "move left 3 meters"},
                        {"role": "assistant", "content": "robot.turn(-90)\nrobot.move_forward(3)"},
                        {"role": "user", "content": "move back and forth between the chair and the table 3 times"},
                        {"role": "assistant", "content": "pos1 = robot.get_pos('chair')\npos2 = robot.get_pos('table')\nfor i in range(3):\n    robot.move_to(pos1)\n    robot.move_to(pos2)"},
                        {"role": "user", "content": "go to the television"},
                        {"role": "assistant", "content": "robot.move_to_object('television')"},
                        {"role": "user", "content": "monitor the chair from 4 view points"},
                        {"role": "assistant", "content": "contour = robot.get_contour('chair')\nnp.random.shuffle(contour)\nfor pos in contour[:4]:\n    robot.move_to(pos)\n    robot.face('chair')"},
                        {"role": "user", "content": "navigate to 3 meters right of the table"},
                        {"role": "assistant", "content": "robot.move_to_right('table')\nrobot.face('table')\nrobot.turn(180)\nrobot.move_forward(3)"},
                        {"role": "user", "content": "turn west"},
                        {"role": "assistant", "content": "robot.turn_absolute(-90)"},
                        {"role": "user", "content": "turn east"},
                        {"role": "assistant", "content": "robot.turn_absolute(90)"},
                        {"role": "user", "content": "turn south"},
                        {"role": "assistant", "content": "robot.turn_absolute(180)"},
                        {"role": "user", "content": "turn north"},
                        {"role": "assistant", "content": "robot.turn_absolute(0)"},
                        {"role": "user", "content": "turn east and then turn left 90 degrees"},
                        {"role": "assistant", "content": "robot.turn_absolute(90)\nrobot.turn(-90)"},
                        {"role": "user", "content": "move 3 meters north of the chair"},
                        {"role": "assistant", "content": "robot.move_north('chair')\nrobot.move_forward(3)"},
                        {"role": "user", "content": "move 1 meters south of the chair"},
                        {"role": "assistant", "content": "robot.move_south('chair')\nrobot.move_forward(3)"},
                        {"role": "user", "content": "move 3 meters west"},
                        {"role": "assistant", "content": "robot.turn_absolute(-90)\nrobot.move_forward(3)"},
                        {"role": "user", "content": lang},
                    ],
                    max_tokens=300
                )
                text = response.choices[0].message.content
                if text:
                    results += text + "\n"
            except Exception as e:
                print(f"Failed to parse instruction: {e}")
                continue

        return results.strip()

    def parse_object_goal_instruction(self, language_instr):
        """解析目标对象指令"""
        try:
            response = self.vlm.chat.completions.create(
                model=self.deployment_name,
                    messages=[
                {
                    "role": "user",
                    "content": "go to the kitchen and then go to the toilet",
                },
                {
                    "role": "assistant",
                    "content": "kitchen, toilet"
                },
                {
                    "role": "user",
                    "content": "go to the chair and then go to another chair"
                },
                {
                    "role": "assistant",
                    "content": "chair, chair"
                },
                {
                    "role": "user",
                    "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting"
                },
                {
                    "role": "assistant",
                    "content": "green sofa, chairs, painting"
                },
                {
                    "role": "user",
                    "content": "approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen"
                },
                {
                    "role": "assistant",
                    "content": "window, television, oven, kitchen"
                },
                {
                    "role": "user",
                    "content": "walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed"
                },
                {
                    "role": "assistant",
                    "content": "plant, table, bedroom, bed"
                },
                {
                    "role": "user",
                    "content": "go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room"
                },
                {
                    "role": "assistant",
                    "content": "stairs, room, book shelf, table, next room"
                },
                {
                    "role": "user",
                    "content": "Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances"
                },
                {
                    "role": "assistant",
                    "content": "table, cushion, column, appliances"
                },
                {
                    "role": "user",
                    "content": "Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table"
                },
                {
                    "role": "assistant",
                    "content": "chair, table"
                },
                {
                    "role": "user",
                    "content": language_instr
                }
            ],
                max_tokens=300
            )
            text = response.choices[0].message.content
            return [x.strip() for x in text.split(",")]
        except Exception as e:
            print(f"Failed to parse object goal instruction: {e}")
            return []


if __name__ == '__main__':
    text = parse_spatial_instruction("go to the sofa, turn right and move in between the table and the chair, and then move back and forth to the keyboard and the screen twice")
    print(text)