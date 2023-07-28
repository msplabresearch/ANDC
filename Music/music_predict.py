import argparse
from smd.data import preprocessing
from smd.data import postprocessing
import smd.utils as utils
import numpy as np
import tensorflow as tf
import keras.models
from tqdm import tqdm
from pathlib import Path
import os, sys
from glob import glob
import shutil
import json










def test_data_processing(file, mean, std):
    if os.path.splitext(file)[1] == '.npy':
        spec = np.load(file)
    else:
        audio = utils.load_audio(file)
        spec = preprocessing.get_spectrogram(audio)
    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    return mels.T


def predict(data_path, output_file, model_path, mean_path, std_path, smoothing):
    mean = np.load(mean_path)
    std = np.load(std_path)

    print("Loading the model " + model_path + "..")
    with tf.device('/cpu:0'):
        model = keras.models.load_model(model_path)
    print("Start the prediction..")

    if os.path.isdir(data_path):
        if output_file != "":
            raise ValueError("It is possible to set an output file only if the input is a file.")

        files = glob.glob(os.path.abspath(data_path) + "/*.npy") + glob.glob(os.path.abspath(data_path) + "/*.wav")
        for file in tqdm(files):
            x = test_data_processing(file, mean, std)
            x = x.reshape((1, x.shape[0], x.shape[1]))
            output = model.predict(x, batch_size=1, verbose=0)[0].T
            output = postprocessing.apply_threshold(output)
            if smoothing:
                output = postprocessing.smooth_output(output)
            annotation = preprocessing.label_to_annotation(output)
            output_path = file.replace(".npy", '') + "_prediction.txt"
            output_path = output_path.replace('.wav','')
            utils.save_annotation(annotation, output_path)
    else:
        file = os.path.abspath(data_path)
        x = test_data_processing(file, mean, std)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        output = model.predict(x, batch_size=1, verbose=0)[0].T
        output = postprocessing.apply_threshold(output)
        if smoothing:
            output = postprocessing.smooth_output(output)
        annotation = preprocessing.label_to_annotation(output)
        if output_file != "":
            output_path = output_file
        else:
            output_path = file.replace(".npy", '') + "_prediction.txt"
            output_path = output_path.replace('.wav','')
        utils.save_annotation(annotation, output_path)








file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[0], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args

args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")




data_path =os.path.join(output_location, "Short_split_file")

output_file =""

model_path = os.path.join(file_dir, "speech-music-detection","checkpoint","weights.28-0.13exp1_blstm.hdf5")

# mean_path = root + "speech-music-detection/checkpoint/mean_gtzan_esc-50_muspeak_musan.npy"
mean_path = os.path.join(file_dir, "speech-music-detection","checkpoint","mean_gtzan_esc-50_muspeak_musan.npy")

# std_path = root + "speech-music-detection/checkpoint/std_gtzan_esc-50_muspeak_musan.npy"
std_path = os.path.join(file_dir, "speech-music-detection","checkpoint","std_gtzan_esc-50_muspeak_musan.npy")

smoothing = True

mean = np.load(mean_path)
std = np.load(std_path)

#load model (keras)
print("Loading the model " + model_path + "..")
with tf.device('/cpu:0'):
    model = keras.models.load_model(model_path)
print("Start the prediction..")


with open(os.path.join(output_location, "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    
    
print("Music/speech inference: ")
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['wav']
    x = test_data_processing(f_path, mean, std)
    x = x.reshape((1, x.shape[0], x.shape[1]))
    output = model.predict(x, batch_size=1, verbose=0)[0].T
    output = postprocessing.apply_threshold(output)
    if smoothing:
        output = postprocessing.smooth_output(output)
    annotation = preprocessing.label_to_annotation(output)
    #make sure its sorted by starting time
    annotation = sorted(annotation, key=lambda x: x[0])
    f_info['speech_music_pred'] = annotation

json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)
