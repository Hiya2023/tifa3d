"""
Post processing of TIFA scores to get average score for each method(model)-aspect pair

"""
import os
import json
from collections import defaultdict

# Folder containing the JSON files (method_aspect_promptname.json)
tifa_result = 'tifa_results'

# Initialization
method_scores_shape = defaultdict(float)
method_counts_shape = defaultdict(int)
prompt_list_shape = defaultdict(list)

method_scores_color = defaultdict(float)
method_counts_color = defaultdict(int)
prompt_list_color = defaultdict(list)


method_scores_style = defaultdict(float)
method_counts_style = defaultdict(int)
prompt_list_style = defaultdict(list)

if os.path.exists("post_processed_files.json") and os.path.getsize("post_processed_files.json")>0:
    with open("post_processed_files.json","r") as f:
        processed_files = set(json.load(f))
else:
    processed_files=set()

# Read and process each file
for filename in os.listdir(tifa_result):
    if not filename.endswith('.json'):
        continue
    if 'questions' in filename.lower():
        continue
    if filename in processed_files:
        continue
    
    processed_files.add(filename)
    
    filepath = os.path.join(tifa_result, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
        method = data.get('method')
        if method == 'mvd_scene':
            continue
        aspect=data.get('aspect')
        prompt=data.get('prompt_name')
        if aspect=='shape':
            avg_score_shape = data.get('average_tifa_score', 0)
            method_scores_shape[method] += avg_score_shape
            method_counts_shape[method] += 1
            prompt_list_shape[method].append(prompt)
        
        if aspect=='color':
            avg_score_color= data.get('average_tifa_score', 0)
            method_scores_color[method] += avg_score_color
            method_counts_color[method] += 1
            prompt_list_color[method].append(prompt)

        if aspect=='style':
            avg_score_style= data.get('average_tifa_score', 0)
            method_scores_style[method] += avg_score_style
            method_counts_style[method] += 1
            prompt_list_style[method].append(prompt)

with open("post_processed_files.json","w") as f:
        json.dump(list(processed_files), f,indent=4)

# Calculate and print average score per method 
for method in method_scores_shape:
    total_score_shape = method_scores_shape[method]
    count_shape = method_counts_shape[method]
    print("count_shape",count_shape)
    prompts_shape= prompt_list_shape[method]
    avg_shape = total_score_shape / count_shape if count_shape else 0
    print(f"{method}: Average TIFA Score for shape aspect = {avg_shape:.4f} ({count_shape} prompts)")
    

print("---------------------------------------------------------------------------")


for method in method_scores_color:
    total_score_color = method_scores_color[method]
    count_color = method_counts_color[method]
    print("count_color",count_color)
    prompts_color = prompt_list_color[method]
    avg_color = total_score_color / count_color if count_color else 0
    print(f"{method}: Average TIFA Score for color aspect = {avg_color:.4f} ({count_color} prompts)")

print("---------------------------------------------------------------------------")


for method in method_scores_style:
    total_score_style = method_scores_style[method]
    count_style = method_counts_style[method]
    print("count_style",count_style)
    prompts_style = prompt_list_style[method]
    avg_style = total_score_style / count_style if count_style else 0
    print(f"{method}: Average TIFA Score for style aspect = {avg_style:.4f} ({count_style} prompts)")


average_scores = {
    "shape":   {m: method_scores_shape[m]  / method_counts_shape[m]  for m in method_scores_shape},
    "color":   {m: method_scores_color[m]  / method_counts_color[m]  for m in method_scores_color},
    "style":   {m: method_scores_style[m]  / method_counts_style[m]  for m in method_scores_style},
}

final_scores = "avg_scores_by_aspect.json"
with open(final_scores, "w") as f:
    json.dump(average_scores, f, indent=4)

print(f"\nSaved per-model averages to “{final_scores}”.")