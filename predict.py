import re
import sys
import base64
import requests
from io import BytesIO
from typing import List, Optional
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
            description="Newline-separated inputs. Can be text strings, base64 data URIs, or image URLs starting with http[s]://",
            default="a\nb",
        ),
        # optional list of direct image files
        image_files: Optional[List[File]] = Input(
            description="List of direct image files for batch processing",
            default=None
        )
    ) -> List[NamedEmbedding]:

        lines = []
        texts = []
        images = []
        image_ids = []  # keep track of 'keys' for each image (url/file/base64)

        # 1) parse lines from the `inputs` string
        for line in inputs.strip().splitlines():
            line = line.strip()
            lines.append(line)

            # handle base64 data: e.g. data:image/png;base64,ABCD...
            if line.startswith("data:image/"):
                try:
                    # strip out prefix like data:image/png;base64,
                    base64_data = re.sub(r"^data:image\/[a-zA-Z]+;base64,", "", line)
                    raw = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(raw))
                    images.append(image)
                    image_ids.append(line)
                except Exception as e:
                    print(f"Failed to decode base64 image: {e}", file=sys.stderr)
            # handle remote image URLs
            elif re.match(r"^https?://", line):
                try:
                    print(f"Downloading {line}", file=sys.stderr)
                    image = Image.open(BytesIO(requests.get(line).content))
                    images.append(image)
                    image_ids.append(line)
                except Exception as e:
                    print(f"Failed to load {line}: {e}", file=sys.stderr)
            else:
                # otherwise, treat it as text
                texts.append(line)

        # 2) parse direct image files if provided
        if image_files:
            for idx, f in enumerate(image_files):
                try:
                    image = Image.open(f)
                    images.append(image)
                    # give each file a unique placeholder ID so we can output it later
                    file_key = f"file_{idx}"
                    image_ids.append(file_key)
                    lines.append(file_key)  # so final output is consistent with text paths
                except Exception as e:
                    print(f"Failed to open file {f.name}: {e}", file=sys.stderr)

        # handle 'None' inputs for processor
        if not images:
            images_for_processor = None
        else:
            images_for_processor = images

        if not texts:
            texts_for_processor = None
        else:
            texts_for_processor = texts

        # 3) run through CLIP processor
        proc_inputs = self.processor(
            text=texts_for_processor,
            images=images_for_processor,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        # 4) get embeddings
        text_outputs = {}
        image_outputs = {}

        if texts_for_processor is not None:
            text_embeds = self.model.get_text_features(
                input_ids=proc_inputs["input_ids"],
                attention_mask=proc_inputs["attention_mask"]
            )
            # map each text to its corresponding embedding
            text_outputs = dict(zip(texts_for_processor, text_embeds))

        if images_for_processor is not None:
            image_embeds = self.model.get_image_features(
                pixel_values=proc_inputs["pixel_values"]
            )
            # map each image key to its corresponding embedding
            image_outputs = dict(zip(image_ids, image_embeds))

        # 5) build outputs in the same order as lines
        outputs = []
        for line in lines:
            # if line was text
            if line in text_outputs:
                outputs.append(NamedEmbedding(
                    input=line,
                    embedding=text_outputs[line].tolist()
                ))
            # otherwise it's an image (url/base64/file)
            elif line in image_outputs:
                outputs.append(NamedEmbedding(
                    input=line,
                    embedding=image_outputs[line].tolist()
                ))
            else:
                # fallback if we missed something
                outputs.append(NamedEmbedding(input=line, embedding=[]))

        return outputs