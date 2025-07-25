"""
TIFA extension for 3d objects (using 12 rendered images from different angles) per prompt for each model-aspect combination
"""
from tifascore import (
    get_question_and_answers, 
    filter_question_and_answers, 
    UnifiedQAModel, 
    tifa_score_single,  
    VQAModel
)
import json
import time
from openai import OpenAI
import os
from statistics import mean  # To compute the average score
from pathlib import Path

processed_prompt_log="processed_prompts.json"
aspects=["shape","color","style"] 

if __name__ == "__main__":
    
    #####################################
    ## TIFA Score Calculation for 12 Images Per Prompt
    #####################################
    
    api_key = os.getenv("OPENAI_API_KEY")
    client=OpenAI(api_key=api_key)
    unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
    vqa_model = VQAModel("mplug-large")
    
    # Root directory for TIFA images
    tifa_root = "../../threestudio_outputs/tifa_images"  
    
    # Load processed prompts
    if os.path.exists(processed_prompt_log):
        with open(processed_prompt_log, "r") as f:
            processed_prompts = json.load(f)
    else:
        processed_prompts={}

    # Output directory for results & questions
    output_dir = "tifa_results"
    os.makedirs(output_dir, exist_ok=True)
  
    # Iterate through all t23d models
    for model in os.listdir(tifa_root):

        model_path = os.path.join(tifa_root, model)

        if not os.path.isdir(model_path):
            continue  # Skip non-directory items
        
        # Iterate through aspects (shape, color, style)
        for aspect in aspects:
            aspect_path = os.path.join(model_path, aspect)
            
            if not os.path.isdir(aspect_path):
                continue
            
            # Iterate through prompts
            for prompt_name in os.listdir(aspect_path):
                prompt_path = os.path.join(aspect_path, prompt_name)

                if not os.path.isdir(prompt_path):
                    continue

                key = f"{model}_{aspect}_{prompt_name}"
                if key in processed_prompts:
                    print(f"Skipping already processed prompt: {key}")
                    continue  # Skip this prompt
            
                print(f"\nProcessing TIFA Score for: {model} -> {aspect} -> {prompt_name}")

                # Generate text description (Assuming prompt_name is the caption)
                text_description = prompt_name.replace("_", " ")  # Modify if needed
                
                # Generate questions using GPT-4o
                gpt4_questions = get_question_and_answers(text_description)
                #print("Generated Questions:", gpt4_questions)
            
                # Save generated questions to a separate JSON
                questions_output_filename = os.path.join(output_dir, f"{model}_{aspect}_{prompt_name}_generated_questions.json")
                
                with open(questions_output_filename, "w") as f:
                    json.dump(gpt4_questions, f, indent=4)
                print(f"Generated questions saved in {questions_output_filename}")

                # Filter questions using UnifiedQA
                filtered_questions = filter_question_and_answers(unifiedqa_model, gpt4_questions)
                print("Filtered Questions:", filtered_questions)

                # Collect TIFA scores for all 12 images
                image_scores = {} #dict with filename-score key-value pair
                question_logs=[] 

                # Ensure correct ordering
                img_files = sorted(
                    [p for p in Path(prompt_path).glob("rgb_*.png")],
                    key=lambda p: int(p.stem.split("_")[1])           # numeric part
                )

                print(f"all image files {img_files}")

                # Process all 12 images
                for img_path in img_files:  
                    img_name=img_path.name
                    print(f"Processing image: {img_path}")

                    #if img_name.startswith("rgb_") and img_name.endswith(".png"):
                        #img_path = os.path.join(prompt_path, img_name)
                        #print(f"Processing image: {img_path}")

                    # Compute TIFA score for this image
                    result = tifa_score_single(vqa_model, filtered_questions, str(img_path))
                    print(f"TIFA Score for {img_name}: {result['tifa_score']}")
                        
                    # Store the individual score
                    image_scores[img_name]=result["tifa_score"]

                     #storing the question-answer details
                    question_logs.append({
                        "image":img_name,
                        "ques_ans":result["question_details"]
                    })
                
                # Compute the final average TIFA score for a given prompt
                average_tifa_score = mean(image_scores.values()) if image_scores else 0
                
                # Print final score
                print(f"\nFinal Average TIFA Score for {prompt_name}: {average_tifa_score}\n")

                # Save results to JSON
                result_dict = {
                    "method": model,
                    "aspect": aspect,
                    "prompt_name": prompt_name,
                    "individual_scores": image_scores,
                    "average_tifa_score": average_tifa_score,
                    "question_logs":question_logs
                }

                # Create output directory
                output_dir = "tifa_results"
                os.makedirs(output_dir, exist_ok=True)

                output_filename = os.path.join(output_dir, f"{model}_{aspect}_{prompt_name}.json")
                with open(output_filename, "w") as f:
                    json.dump(result_dict, f, indent=4)

                print(f"Results saved in {output_filename}")
            
                # **Mark this prompt as processed**
                processed_prompts[key] = True

                # **Save the updated log file**
                with open(processed_prompt_log, "w") as f:
                    json.dump(processed_prompts, f, indent=4)

                print(f"Updated processed log: {processed_prompt_log}")
