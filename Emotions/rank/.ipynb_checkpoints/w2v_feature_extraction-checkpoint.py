import soundfile as sf
from scipy import signal
from tqdm import tqdm
import torch
import json
from datasets import load_dataset
import soundfile as sf
from scipy.io import savemat
from transformers import AutoProcessor, AutoModelForPreTraining
from pathlib import Path
import warnings
import os, sys
import json
import numpy as np
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")





#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args



args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    
    



out_path = os.path.join(output_location, "Short_split_w2v_base_feats")

if not os.path.exists(out_path):
    os.mkdir(out_path)    
    
print("Extracting wav2vec features: ")
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['wav']
    audio_input, sample_rate = sf.read(f_path)
    if len(audio_input.shape) == 2:
        audio_input = audio_input[:,0]
    if sample_rate != 16000:
        audio_input = signal.resample(audio_input, int((len(audio_input)/sample_rate)*16000))
        sample_rate = 16000

    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    logits = model(input_values)
    logits = logits.projected_states.detach().numpy()
    
    save_path = os.path.join(out_path, filename+'.npy')
    np.save(save_path, logits)
    f_info['filepaths']['w2v_base_feats'] = save_path
    

print("Writing json file")
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)    