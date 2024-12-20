import sys
import requests
import torch
from typing import List, Union
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cog import BasePredictor, Path, Input, BaseModel, File

class NamedEmbedding(BaseModel):
    input: str
    embedding: List[float]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = CLIPModel.from_pretrained("/weights", local_files_only=True).to("cuda")
        self.processor = CLIPProcessor.from_pretrained("/weights", local_files_only=True)

    def predict(
        self,
        inputs: str = Input(
            description="Newline-separated inputs. Can be text strings, image files, or image URLs starting with http[s]://",
            default="a\nb",
        ),
        image_file: File = Input(
            description="Direct image file input",
            default=None
        )
    ) -> List[NamedEmbedding]:
        lines = []
        texts = []
        images = []
        image_identifiers = []  # Store either URLs or file paths

        # Handle direct file input first
        if image_file:
            image = Image.open(image_file)
            images.append(image)
            image_identifiers.append(str(image_file))
            lines.append(str(image_file))

        # Process newline-separated inputs
        for line in inputs.strip().splitlines():
            line = line.strip()
            lines.append(line)
            
            if line.startswith(('http://', 'https://')):
                try:
                    print(f"Downloading {line}", file=sys.stderr)
                    image = Image.open(BytesIO(requests.get(line).content))
                    images.append(image)
                    image_identifiers.append(line)
                except Exception as e:
                    print(f"Failed to load {line}: {e}", file=sys.stderr)
            elif Path(line).exists():  # Check if input is a valid file path
                try:
                    image = Image.open(line)
                    images.append(image)
                    image_identifiers.append(line)
                except Exception as e:
                    print(f"Failed to load file {line}: {e}", file=sys.stderr)
                    texts.append(line)  # Fallback to treating as text
            else:
                texts.append(line)

        # Process inputs through CLIP
        if not images:
            images = None
        if not texts:
            texts = None

        # Process texts and images separately to handle padding correctly
        text_inputs = None
        if texts:
            text_inputs = self.processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP's default max length
            ).to("cuda")

        image_inputs = None
        if images:
            # Process images one by one to ensure consistent tensors
            processed_images = []
            for img in images:
                processed = self.processor.feature_extractor(
                    img, 
                    return_tensors="pt"
                )
                processed_images.append(processed["pixel_values"][0])
            
            if processed_images:
                image_inputs = {
                    "pixel_values": torch.stack(processed_images).to("cuda")
                }

        # Get embeddings
        text_outputs = {}
        if text_inputs:
            text_embeds = self.model.get_text_features(**text_inputs)
            text_outputs = dict(zip(texts, text_embeds))

        image_outputs = {}
        if image_inputs:
            image_embeds = self.model.get_image_features(**image_inputs)
            image_outputs = dict(zip(image_identifiers, image_embeds))

        # Construct output
        outputs = []
        for line in lines:
            if line in text_outputs:
                outputs.append(
                    NamedEmbedding(input=line, embedding=text_outputs[line].tolist())
                )
            elif line in image_outputs:
                outputs.append(
                    NamedEmbedding(input=line, embedding=image_outputs[line].tolist())
                )

        return outputs