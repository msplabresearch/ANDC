import os, sys
from pathlib import Path
import json
import shutil


#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args

def create_dir(dir_path):
    if os.path.exists(dir_path):
        print("Directory {} already exists".format(dir_path))
    else:
        print("Creating directory {}".format(dir_path))
        os.mkdir(dir_path)
        

if __name__ == "__main__":
    print("hello", sys.argv)
    args = get_args()

    
    root = args.root
    output_path = os.path.join(root, "Outputs")
    mfa_root = os.path.join(output_path, 'mfa_corpus')
    mfa_speaker = os.path.join(mfa_root, 'speaker1')

    
    create_dir(mfa_root)
    create_dir(mfa_speaker)

    with open(os.path.join(output_path, "Long_files.json"), "r") as openfile:
        long_files = json.load(openfile)



    for audioname, val in long_files.items():
        filepath = val['filepaths']['wav']
        #copy audio file to mfa dir
        shutil.copy(filepath, mfa_speaker)

        #create text file for mfa 
        dest_file = os.path.join(mfa_speaker, audioname+'.txt')
        text = val['text']
        with open(dest_file, "w") as text_file:
            text_file.write(text)