import json
from .openai_api import openai_completion
from tqdm import tqdm
import random

categories = ['object', 'human', 'animal', 'food', 'activity', 'attribute', 'counting', 'color', 'material', 'spatial', 'location', 'shape', 'other']

prompt = """
Given image descriptions, generate one or two multiple-choice questions that verifies if the image description is correct.
Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other), and then generate a question for each type.

Description: A red car.
Entities: car
Activities:
Colors: red
Counting:
Other attributes:
Questions and answers are below:
About car (object):
Q: is this a car?
Choices: yes, no
A: yes
Q: what type of vehicle is this?
Choices: car, truck, motorcycle, bicycle
A: car
About red (color):
Q: is the car red?
Choices: yes, no
A: yes
Q: what color is the car?
Choices: red, blue, black, white
A: red

Description: A white pillow.
Entities: pillow
Activities:
Colors: white
Counting:
Other attributes:
Questions and answers are below:
About pillow (object):
Q: is this a pillow?
Choices: yes, no
A: yes
Q: what household item is this?
Choices: pillow, blanket, chair, table
A: pillow
About white (color):
Q: is the pillow white?
Choices: yes, no
A: yes
Q: what color is the pillow?
Choices: white, black, blue, red
A: white

Description: A blue and white striped shirt.
Entities: shirt
Activities:
Colors: blue, white
Counting:
Other attributes: striped
Questions and answers are below:
About shirt (object):
Q: is this a shirt?
Choices: yes, no
A: yes
Q: what piece of clothing is shown?
Choices: shirt, pants, jacket, hat
A: shirt
About blue and white (color):
Q: does the shirt have blue and white colors?
Choices: yes, no
A: yes
Q: what colors are on the shirt?
Choices: blue and white, red and black, green and yellow, purple and orange
A: blue and white
About striped (attribute):
Q: is the shirt striped?
Choices: yes, no
A: yes
Q: what pattern does the shirt have?
Choices: striped, polka dot, plain, checked
A: striped

Description: A silver laptop with a blue case.
Entities: laptop, case
Activities:
Colors: silver, blue
Counting:
Other attributes:
Questions and answers are below:
About laptop (object):
Q: is this a laptop?
Choices: yes, no
A: yes
Q: what type of device is shown?
Choices: laptop, tablet, desktop, smartphone
A: laptop
About blue case (object):
Q: does the laptop have a case?
Choices: yes, no
A: yes
Q: what color is the laptop case?
Choices: blue, red, black, white
A: blue
About silver (color):
Q: is the laptop silver?
Choices: yes, no
A: yes
Q: what is the color of the laptop body?
Choices: silver, gold, bronze, copper
A: silver

Description: A round ball.
Entities: ball
Activities:
Colors:
Shapes: round
Other attributes:
Questions and answers are below:
About ball (object):
Q: is this a ball?
Choices: yes, no
A: yes
Q: what object is this?
Choices: ball, cube, cone, cylinder
A: ball
About round (shape):
Q: is the ball round?
Choices: yes, no
A: yes
Q: what shape is the ball?
Choices: round, square, triangular, rectangular
A: round

Description: A square table.
Entities: table
Activities:
Colors:
Shapes:square
Other attributes:
Questions and answers are below:
About table (object):
Q: is this a table?
Choices: yes, no
A: yes
Q: what piece of furniture is shown?
Choices: table, chair, sofa, desk
A: table
About square (shape):
Q: is the table square?
Choices: yes, no
A: yes
Q: what is the shape of the table?
Choices: square, round, oval, rectangular
A: square

Description: A lamp with a conical shade and a circular base.
Entities: lamp, shade, base
Activities:
Colors:
Shapes:conical, circular
Other attributes: 
Questions and answers are below:
About lamp (object):
Q: is this a lamp?
Choices: yes, no
A: yes
Q: what type of item is this?
Choices: lamp, clock, radio, chair
A: lamp
About conical shade (shape):
Q: does the lamp have a conical shade?
Choices: yes, no
A: yes
Q: what is the shape of the lamp shade?
Choices: conical, cylindrical, rectangular, spherical
A: conical
About circular base (shape):
Q: does the lamp have a circular base?
Choices: yes, no
A: yes
Q: what shape is the base of the lamp?
Choices: circular, square, triangular, oval
A: circular

Description: A dining table with an elliptical top and tapered legs.
Entities: table, legs
Activities:
Colors:
Counting:
Other attributes: elliptical, tapered
Questions and answers are below:
About dining table (object):
Q: is this a dining table?
Choices: yes, no
A: yes
Q: what type of furniture is this?
Choices: dining table, coffee table, desk, bed
A: dining table
About elliptical top (attribute):
Q: does the table have an elliptical top?
Choices: yes, no
A: yes
Q: what is the shape of the table top?
Choices: elliptical, rectangular, square, circular
A: elliptical
About tapered legs (attribute):
Q: are the table legs tapered?
Choices: yes, no
A: yes
Q: what is unique about the table legs?
Choices: tapered, straight, curved, chunky
A: tapered

Description: A cat sits on a sofa.
Entities: cat, sofa
Activities: sits
Spatial Relation: on
Questions and answers are below:
About cat (animal):
Q: Is there a cat?
Choices: yes, no
A: yes
Q: What animal is shown?
Choices: cat, dog, bird, fish
A: cat
About sofa (object):
Q: Is there a sofa?
Choices: yes, no
A: yes
Q: What piece of furniture is in the image?
Choices: sofa, chair, table, bed
A: sofa
About on (spatial):
Q: Is the cat on the sofa?
Choices: yes, no
A: yes
Q: What is the spatial relation between the cat and the sofa?
Choices: on, under, beside, behind
A: on

Description: A red ball is under a chair.
Entities: red ball, chair
Activities: is
Spatial Relation: under
Questions and answers are below:
About red ball (object):
Q: Is there a ball?
Choices: yes, no
A: yes
Q: What color is the ball?
Choices: red, blue, green, yellow
A: red
About chair (object):
Q: Is there a chair?
Choices: yes, no
A: yes
About under (spatial):
Q: Is the ball under the chair?
Choices: yes, no
A: yes
Q: What is the spatial relation between the ball and the chair?
Choices: under, on, beside, above
A: under

Description: A blue vase is on a green table.
Entities: blue vase, green table
Activities: is
Spatial Relation: on
Questions and answers are below:
About blue vase (object):
Q: Is there a vase?
Choices: yes, no
A: yes
Q: What color is the vase?
Choices: blue, red, white, black
A: blue
About green table (object):
Q: Is there a table?
Choices: yes, no
A: yes
Q: What color is the table?
Choices: green, blue, red, yellow
A: green
About on (spatial):
Q: Is the vase on the table?
Choices: yes, no
A: yes
Q: What is the spatial relation between the vase and the table?
Choices: on, under, beside, behind
A: on

Description: A dog is beside a tree.
Entities: dog, tree
Activities: is
Spatial Relation: beside
Questions and answers are below:
About dog (animal):
Q: Is there a dog?
Choices: yes, no
A: yes
Q: What animal is shown?
Choices: dog, cat, bird, horse
A: dog
About tree (object):
Q: Is there a tree?
Choices: yes, no
A: yes
Q: What natural object is present?
Choices: tree, bush, flower, rock
A: tree
About beside (spatial):
Q: Is the dog beside the tree?
Choices: yes, no
A: yes
Q: What is the dog's position relative to the tree?
Choices: beside, under, on, behind
A: beside

Description: A rustic Italian teapot. 
Entities: teapot 
Activities: 
Colors: 
Shapes: 
Other attributes: rustic, Italian
Questions and answers are below: 
About teapot (object): 
Q: Is this a teapot? 
Choices: yes, no 
A: yes 
Q: What type of object is shown? 
Choices: teapot, vase, coffee mug, kettle
A: teapot 
About rustic (attribute): 
Q: Is the teapot rustic? 
Choices: yes, no 
A: yes 
Q: How would you describe the teapot’s style? 
Choices: rustic, modern, futuristic, minimalistic 
A: rustic 
About Italian (attribute): 
Q: Is the teapot Italian? 
Choices: yes, no 
A: yes 
Q: What is the cultural style of the teapot? 
Choices: Italian, Japanese, Chinese, American 
A: Italian

Description: A minimalist Japanese vase with floral engravings. 
Entities: vase 
Activities: 
Colors: 
Shapes: 
Other attributes: minimalist, Japanese, floral engravings 
Questions and answers are below: 
About vase (object): 
Q: Is this a vase? 
Choices: yes, no 
A: yes 
Q: What type of item is shown? 
Choices: vase, teapot, jar, bottle 
A: vase 
About minimalist (attribute): 
Q: Is the vase minimalist? 
Choices: yes, no 
A: yes 
Q: How would you describe the style of the vase? 
Choices: minimalist, baroque, victorian, bohemian 
A: minimalist 
About Japanese (attribute): 
Q: Is the vase Japanese? 
Choices: yes, no A: yes 
Q: What is the cultural style of the vase? 
Choices: Japanese, Italian, African, Swedish 
A: Japanese 
About floral engravings (attribute): 
Q: Does the vase have floral engravings? 
Choices: yes, no 
A: yes 
Q: What kind of engravings does the vase have? 
Choices: floral, abstract, geometric, striped 
A: floral

Description: A rustic Indian wooden cabinet beside a minimalist Japanese painting. 
Entities: cabinet, painting 
Activities: 
Colors: 
Shapes: 
Spatial Relation: beside 
Other attributes: rustic, Indian, wooden, minimalist, Japanese 
Questions and answers are below: 
About cabinet (object): 
Q: Is there a cabinet? 
Choices: yes, no 
A: yes 
Q: What piece of furniture is shown? 
Choices: cabinet, table, chair, bed 
A: cabinet 
About painting (object): 
Q: Is there a painting? 
Choices: yes, no 
A: yes 
Q: What decorative item is present? 
Choices: painting, sculpture, poster, tapestry 
A: painting 
About rustic (attribute): 
Q: Is the cabinet rustic? 
Choices: yes, no 
A: yes 
Q: How would you describe the cabinet’s style? 
Choices: rustic, modern, minimalistic, futuristic 
A: rustic 
About Indian (attribute): 
Q: Is the cabinet Indian? 
Choices: yes, no 
A: yes 
Q: What is the cultural style of the cabinet? 
Choices: Indian, Japanese, Italian, Greek 
A: Indian 
About wooden (material): 
Q: Is the cabinet made of wood? 
Choices: yes, no 
A: yes 
Q: What is the cabinet’s material? 
Choices: wood, metal, plastic, glass 
A: wood 
About minimalist (attribute): Q: Is the painting minimalist? 
Choices: yes, no 
A: yes 
Q: How would you describe the painting’s style? 
Choices: minimalist, baroque, victorian, bohemian 
A: minimalist 
About Japanese (attribute): 
Q: Is the painting Japanese? 
Choices: yes, no 
A: yes 
Q: What is the cultural style of the painting? 
Choices: Japanese, Indian, Italian, African 
A: Japanese 
About beside (spatial): 
Q: Is the cabinet beside the painting? 
Choices: yes, no 
A: yes 
Q: What is the spatial relation between the cabinet and the painting? 
Choices: beside, above, behind, under 
A: beside

Description: """

def parse_resp(resp):
    resp = resp.split('\n')
    
    question_instances = []
    
    this_entity = None
    this_type = None
    this_question = None
    this_choices = None
    this_answer = None
    
    for line_number in range(6, len(resp)):
        line = resp[line_number]
        print("line",line)
        if line.startswith('About '):
            whole_line = line[len('About '):-1]
            this_entity = whole_line.split(' (')[0]
            this_type = whole_line.split(' (')[1].split(')')[0]
            
        elif line.startswith('Q: '):
            this_question = line[3:].strip()
        elif line.startswith('Choices: '):
            this_choices = [choice.strip() for choice in line[9:].split(',')]
            #this_choices = line[9:].split(', ')
        elif line.startswith('A: '):
            #this_answer = line[3:]
            this_answer = line[3:].strip()
            
            if this_entity and this_question and this_choices:
                question_instances.append((this_entity, this_question, this_choices, this_answer, this_type))
            this_question = None
            this_choices = None
            this_answer = None
            
    return question_instances


## Generate questions for a caption with GPT-3

def get_question_and_answers(caption):
    print(caption, "initial")
   
    this_prompt = prompt + caption
    # this_prompt = prompt + f"{caption}\nEntities:"
    #print(this_prompt)
    resp = openai_completion(this_prompt)
    if resp:
     print("API Response:", resp)
    else:
     print("Failed to get a response from the OpenAI API.")
    
    with open('resp.json', 'w') as f:
        json.dump(resp, f)
    
    question_instances = parse_resp(resp)
    print(question_instances, "ques insta")
    this_caption_qas = []
    
    for question_instance in question_instances:
        this_qa = {}
        this_qa['caption'] = caption
        this_qa['element'] = question_instance[0]
        this_qa['question'] = question_instance[1]
        this_qa['choices'] = question_instance[2]
        this_qa['answer'] = question_instance[3]
        this_qa['element_type'] = question_instance[4]
        
        if question_instance[4] not in categories:
            continue
            
        if this_qa['element_type'] in ['animal', 'human']:
            this_qa['element_type'] = 'animal/human'
            
        this_caption_qas.append(this_qa)
        
    return this_caption_qas
