import os
from huggingface_hub._login import _login



def logging_in():
    _login(token=os.environ['HUGGING_FACE_TOKEN'], add_to_git_credential=False)