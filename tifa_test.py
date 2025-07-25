from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_benchmark, tifa_score_single,  VQAModel
import json
import openai

if __name__ == "__main__":
    
    #####################################
    ## Test TIFA score on benchmark
    #####################################
    
    # test tifa benchmarking
    #results = tifa_score_benchmark("mplug-large", "sample/sample_question_answers.json", "sample/sample_imgs.json")
    #with open("sample/sample_evaluation_result.json", "w") as f:
        #json.dump(results, f, indent=4)
    
    
    #####################################
    ## Test TIFA score on one image
    #####################################
    
    # prepare the models
    unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
    vqa_model = VQAModel("mplug-large")
    
    img_path = "sample/sparrow_15.png"
    text = "A red and white sparrow."
    
    # Generate questions with GPT-4o-mini (modified from 3.5 turbo)
    gpt4_questions = get_question_and_answers(text)
    print("ques using gpt4", gpt4_questions)
    # Filter questions with UnifiedQA
    filtered_questions = filter_question_and_answers(unifiedqa_model, gpt4_questions)
    
    # See the questions
    print(filtered_questions)

    # calucluate TIFA score
    result = tifa_score_single(vqa_model, filtered_questions, img_path)
    print(f"TIFA score is {result['tifa_score']}")
    print(result)
    
    
    
