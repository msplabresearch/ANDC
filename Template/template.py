import os, sys
import json
from tqdm import tqdm
from pathlib import Path



#import ANDC general arguments
#change parents index based on location of the directory
#file_dir links to /ANDC utils_dir links to /ANDC/utils
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args



#get args (see ANDC/utils for file)
args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")


#read Short_files.json which should contains all required information reguarding audio samples
with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)

    
#loop through the audio_files, inferencing, and saving the inference in f_info
print("Doing X thing: ")
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['wav']

    
    #change my_model to custom model 
    preds = my_model(f_path)
    
    #save predictions
    f_info['My_model_preds'] = preds
    

#save the json file
print("Writing json file")
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)    