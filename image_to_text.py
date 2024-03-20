from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
#change to float16 if on Nvidia gpu
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
#uncomment if on Nvidia gpu
#pipe = pipe.to("cuda")

prompt = "a photo of a hippo riding a monkey on saturn's rings"
image = pipe(prompt).images[0]  
    
image.save("new.png")