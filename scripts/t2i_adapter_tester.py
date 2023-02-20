import cv2
import argparse
import torch
import sys
import os
root_path = os.getcwd()
sys.path.append(f"{root_path}/src")
import diffusers
from PIL import Image
from diffusers import StableDiffusionT2IAdapterPipeline, DPMSolverMultistepScheduler
from extra.t2iadapter.adapter import Adapter
from basicsr.utils import img2tensor, tensor2img, scandir, get_time_str, get_root_logger, get_env_info

diffusers.utils.logging.disable_progress_bar()


class DummySafetyChecker():
    def safety_checker(self, images, *args, **kwargs):
        return images, [False] * len(images)


def loadmodel(pipeline_name, model_path, **kwargs):
    print("load pipeline")
    print("load model from:", pipeline_name, model_path)

    kwargs["torch_dtype"] = torch.float16
    kwargs["revision"] = "fp16"

    safechecker = DummySafetyChecker().safety_checker
    kwargs["safety_checker"] = safechecker

    pipe = StableDiffusionT2IAdapterPipeline.from_pretrained(model_path, **kwargs)
    return pipe.to("cuda")


def generation(pipe, prompt, seed, features_adapter=None):
    settings = {
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
    }
    settings["prompt"] = prompt
    settings["features_adapter"] = features_adapter
    settings["features_adapter_strength"] = 0.5
    images = pipe(**settings).images
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description="auto aiart generator")
    parser.add_argument(
        "-p", "--pipeline", help="Diffusers pipeline name", required=True
    )
    parser.add_argument("-m", "--model_path", help="model path", required=True)
    parser.add_argument("-ad", "--ckpt_ad", help="path to checkpoint of adapter", required=True)
    parser.add_argument("-cond", "--path_cond", help="path to adapter condition", required=True)
    args = parser.parse_args()
    kwargs = {}

    device = "cuda"
    model_ad = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device).half()
    model_ad.load_state_dict(torch.load(args.ckpt_ad))
    edge = cv2.imread(args.path_cond)
    edge = cv2.resize(edge,(512,512))
    edge = img2tensor(edge)[0].unsqueeze(0).unsqueeze(0)/255.
    edge = edge>0.5
    edge = edge.float().half()
    features_adapter = model_ad(edge.to(device))

    pipe = loadmodel(args.pipeline, args.model_path, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    prompt = f"A car with flying wings"
    outputimg = generation(pipe, prompt, 52, features_adapter)
    filename = f"output.png"
    outputimg[0].save(f"{filename}")

if __name__ == "__main__":
    sys.exit(main())  # next section explains the use of sys.exit
