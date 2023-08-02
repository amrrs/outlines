"""Module that contains all the models integated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""
from . import embeddings, image_generation, text_completion
from .ctranslate2 import ctranslate2
from .hf_diffusers import HuggingFaceDiffuser
from .hf_transformers import HuggingFaceCompletion
from .openai import OpenAICompletion, OpenAIEmbeddings, OpenAIImageGeneration
from .transformers import transformers
