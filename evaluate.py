import argparse
import logging
import numpy as np
import argparse
from PIL import Image
from veagle.models import load_model_and_preprocess
from download_hf import download_models

def check_model_availability():
    """Checks models availability
    """
    logging.info("Checking the availability of the models")
    download_models()
    logging.info("Models are available")    
    
    
def disable_torch_init():
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        
def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for Evaluation")
    parser.add_argument(
        "--answer_mc",
        action="store_true",
        default=False,
        help="Whether to evaluate multiple choice question with candidates."
    )
    parser.add_argument(    
        "--answer_qs",
        action="store_true",
        default=False,
        help="Whether to evaluate only one question image."
    )

    parser.add_argument("--model_name", type=str, default="bliva_vicuna")
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify which gpu device to use.")
    parser.add_argument("--img_path", type=str, required=True, help="the path to the image")
    parser.add_argument("--question", type=str, required=True,  help="the question to ask")
    parser.add_argument("--candidates", type=str,  help="list of choices for mulitple choice question")

    args = parser.parse_args()
    return args

def eval_one(image, question, model):
    """
    Evaluate one question
    """
    outputs = model.generate({"image": image, "prompt": question})
    print("=====================================")
    print("Question:", question[0])
    print("-------------------------------------")
    print("Outputs: ", outputs[0])


def eval_candidates(image, question, candidates, model):
    """
    Evaluate with candidates
    """
    outputs = model.predict_class({"image": image, "prompt": question}, candidates)
    print("=====================================")
    print("Question:", question[0])
    print("-------------------------------------")
    print("Candidates:", candidates)
    print("-------------------------------------")
    print("Outputs: ", candidates[outputs[0][0]])



def main(args):
    np.random.seed(0)
         
    disable_torch_init()
    
    # check model availability
    check_model_availability()
    
    if args.model_name == "veagle_mistral":
        model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type="mistral7b", is_eval=True, device=args.device)
    vis_processor = vis_processors["eval"]
    
    image = Image.open(args.img_path).convert('RGB')

    question = [args.question]
    
    image = vis_processor(image).unsqueeze(0).to(args.device)
   
    if args.answer_qs:
        eval_one(image, question, model)
    elif args.answer_mc:
        candidates = [candidate.strip() for candidate in args.candidates.split(",")]
        eval_candidates(image, question, candidates, model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
