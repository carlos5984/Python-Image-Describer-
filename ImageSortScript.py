from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from easyocr import Reader
import numpy as np
import json
import os
import pandas as pd

reader = Reader(['en', 'es', 'pt'], gpu=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
imagefolder = "images"
print("vivo")

images = {}
def GenDescriptionAndGetText(img):
    if isinstance(img, str):
        file_name = img.split("\\")[-1]
        img = Image.open(img).convert("RGB")
        img = np.array(img)
    elif isinstance(img, np.ndarray):
        file_name = "uploaded_image"
    else:
        raise ValueError("Unknown input type for image!")
    max_size= 600
    #reescaala
    if (max(img.shape)> max_size):
        scale = max_size / max(img.shape)
        new_width, new_height = int(img.shape[1] * scale), int(img.shape[0] * scale)
        img = np.array(Image.fromarray(img).resize((new_width,new_height), Image.Resampling.NEAREST))
    #extrai o texto
    outputtext = reader.readtext(img)
    extractedtext = []
    for (_, text, _) in outputtext:
        extractedtext.append(text)
    #extrai descricao
    imginput = Image.fromarray(img)
    inputs = processor(imginput ,return_tensors="pt").to(device)
    outputdesc = model.generate(**inputs)
    description = processor.decode(outputdesc[0], skip_special_tokens = True)
    images[file_name] = [description,extractedtext]
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=2)
    return  file_name, description, extractedtext

rows = []

for filename in os.listdir(imagefolder):
    print('tentando')
    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        image_path = os.path.join(imagefolder, filename)
        file_name, description, extracted_texts = GenDescriptionAndGetText(image_path)
        #salva cada informacao em cada coluna do ngc do excel
        rows.append({
            "Filename": file_name,
            "Description": description,
            "Extracted Text": " | ".join(extracted_texts)
        })
        

df = pd.DataFrame(rows)
df.to_excel("output.xlsx", index=False)
print("WOOOOOOOOOOOOOOOOOOOO")