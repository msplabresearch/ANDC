#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, sys
import numpy as np
from scipy.io import savemat
import subprocess
import json
from glob import glob
from pathlib import Path
import shutil
from tqdm import tqdm



#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args



args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(output_location, "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    
    
OPENSMILE_PATH = args.opensmile



def SmileExtract(input_path, out_path, audio_files, feature_type, desc_type = 'lld'):
    ERROR_record = ''
    for filename, f_info in tqdm(audio_files.items()):
        f_path = f_info['filepaths']['wav']
        if '.wav' in f_path:
            input_file = f_path
            output_file = os.path.join(out_path, filename + '.arff')
            f_info['filepaths']['opensmile_'+desc_type] = output_file

            if desc_type == 'lld':
                lld_location = os.path.join(file_dir, "opensmile-2.3.0_lld","IS13_ComParE.conf")
                cmd = OPENSMILE_PATH+' -l 1 -C '+lld_location+' -I ' + input_file + ' -O ' + output_file
            elif desc_type == 'hld':
                hld_location = os.path.join(file_dir, "opensmile-2.3.0_hld","IS13_ComParE.conf")
                cmd = OPENSMILE_PATH+' -l 1 -C '+hld_location+' -I ' + input_file + ' -O ' + output_file
            else:
                raise ValueError('Unsupport desc_type Type!')
            try:
                subprocess.call(cmd, shell=True)
            except:
                ERROR_record += 'Error: '+fileNames[i]+'\n'

        else:
            ERROR_record += 'Source not WAV file: ' +fileNames[i]+'\n'
    print(ERROR_record)
    
    
    

    
# feature_arff to feature_mat
def TryToFloat(single_data):
    try:
        return float(single_data)
    except:
        return None

def LoadFeature(filename):
    content = open(filename, 'r').read()
    data = content.split('@data\n')[1].split('\n')
    data = filter(None, data)
    feature = [[TryToFloat(data_split) for data_split in d.split(',') \
                if TryToFloat(data_split)!=None] for d in data]
    return feature

with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    
    

####################### eGeMAPS-LLD ####################
feat = 'OpenSmile_lld_IS13ComParE'  # Dim=130 LLDs

# Part 1: OpenSmile => Arff File
input_path = os.path.join(output_location, "Short_split_file")
out_path = os.path.join(output_location, "Short_split_"+feat)
if not os.path.exists(out_path):
    os.mkdir(out_path)

print("Running openSMILE lld  extractor")
SmileExtract(input_path, out_path, audio_files, feature_type='IS13ComParE', desc_type='lld')


feat = 'OpenSmile_hld_IS13ComParE' # Dim=6373 HLDs
input_path = os.path.join(output_location, "Short_split_file")
out_path = os.path.join(output_location, "Short_split_"+feat)
if not os.path.exists(out_path):
    os.mkdir(out_path)
print("Running openSMILE hld extractor")
SmileExtract(input_path, out_path, audio_files, feature_type='IS13ComParE', desc_type='hld') 



json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)