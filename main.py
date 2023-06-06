from typing import Dict,Union
from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client
import os
from fastapi.responses import FileResponse
from diffusers import DiffusionPipeline
app = FastAPI()

# # Load the Hugging Face model
# model_name = "runwayml/stable-diffusion-v1-5"
# model = Diffusion.from_pretrained(model_name)

# SDK initialization

# from imagekitio import ImageKit
# imagekit = ImageKit(
#     private_key='private_uMtwegvlzGp8qapJdprjb2t8e+8=',
#     public_key='public_lv0mfZTFKTeR7M2AgkQuzxCyIxk=',
#     url_endpoint='https://ik.imagekit.io/fpzf5h2jb'
# )

class TextInput(BaseModel):
    text: str


class ImageOutput(BaseModel):
    image_link: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# @app.post("/generate_image", response_model=ImageOutput)
# def generate_image(input_data: TextInput) -> Dict[str, str]:
#     # Generate the image
#     # output = model.generate(text=input_data.text)

#     # Convert the image to a link or any other desired output format
#     # image_link = "https://example.com/path/to/image.jpg"
#     client = Client("https://vedant20-runwayml-stable-diffusion-v1-5.hf.space/")
#     result = client.predict(
#                     input_data,	# str  in 'Input' Textbox component
#                     api_name="/predict"
#     )
#     print("hitting")
#     return {"image_link": result}


app = FastAPI()
client = Client.duplicate("runwayml/stable-diffusion-v1-5", hf_token="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") 
# client = Client("https://vedant20-runwayml-stable-diffusion-v1-5.hf.space/")


class ImageText(BaseModel):
    image_text: str


@app.post("/predict")
def predict_image_link(image: ImageText):
    result = client.predict(image.image_text, api_name="/predict")

    # imagekit_response = imagekit.upload_file(
    #     file=url,
    #     file_name="test-url.jpg",
    #     options=UploadFileRequestOptions(
    #         response_fields=["is_private_file", "tags"],
    #         tags=["tag1", "tag2"]
    #     )
    # image_url = imagekit_response["response"]["url"]
    # image_link = result["image_link"]
    # print(imagekit_response)

    # output_file = io.BytesIO(result)
    res =  FileResponse(result, media_type="image/jpeg")
    # os.remove(result)
    return res
    # return {"image_link": image_url}


@app.post("/model")
def predict_by_model(image: ImageText):
    # result = client.predict(image.image_text, api_name="/predict")
    # res =  FileResponse(result, media_type="image/jpeg")
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("mps")

    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    prompt = image.image_text
    # First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
    _ = pipe(prompt, num_inference_steps=1)

    # Results match those from the CPU device after the warmup pass.
    imageRes = pipe(prompt).images[0]
    print(imageRes)
    # os.remove(result)
    return imageRes





# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# pipe = pipe.to("mps")

# # Recommended if your computer has < 64 GB of RAM
# pipe.enable_attention_slicing()

# prompt = "a photo of an astronaut riding a horse on mars"
# # First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
# _ = pipe(prompt, num_inference_steps=1)

# # Results match those from the CPU device after the warmup pass.
# image = pipe(prompt).images[0]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)