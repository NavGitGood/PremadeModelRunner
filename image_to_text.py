from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
#pipe = pipe.to("cuda")

prompt = "a photo of a hippo riding a monkey on saturn's rings"
image = pipe(prompt).images[0]  
    
image.save("new.png")