

# Suggested Environment and Requirements
1. Python 3.8.10 or above
2. Ubuntu 20.04
3. keras version 2.7.0 or above
4. tensorflow version 2.7.0 or above
5. PyTorch 1.9.0 or above
6. CUDA 11.6 or above
7. The scipy, numpy, transformers, soundfile and pandas packages
8. The model is trained using MSP-Podcast corpus (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html))

# How to use the trained model 

1. Place the audio files in 'Audios' directory (we place some sample wav files to make the code run).
2. Generate file_names.txt file containing names of all audio files present in Audios directory.
3. First compile:
   * python3 Test_feature_extraction.py 
   This will extract Wav2vec2-large-robust features. 
4. To test any model run: 
   * python3 Testing.py
   Make sure to copy the appropriate model (Arousal (model ending with aro_weights.h5), Valence (model ending with val_weights.h5), for dominance (model ending with dom_weights.h5))
5. This will generate test_values.txt, which contains wav file name and corresponding Arousal or Valence or Dominance value (scale 1 to 7).
6. comment line 8 in lstm_testing.py (os.environ["CUDA_VISIBLE_DEVICES"]="0") if GPU is not available.
7. Make sure each audio file is between 3 sec to 11 sec duration.
8. For training any model and extract wav2vec2-large-robust features using Test_feature_extraction.py then run:
   * python3 lstm_training.py -ep xx -batch xx -emo xxx(aro, val, or dom) -atten RnnAttenVec



# Before running make sure all the below libraries are installed or imported.

==============================  <br />
import numpy as np <br />
import os <br />
import pandas as pd <br />
from keras.models import Model <br />
from keras import backend as K <br />
from tensorflow.keras.layers import BatchNormalization <br />
from keras.layers import LSTM, Input, Multiply <br />
import random <br />
from utils import getPaths_test, DynamicChunkSplitTestingData, evaluation_metrics <br />
from model_utils import mean, reshape <br />
from model_utils import atten_gated, atten_rnn, atten_selfMH, output_net <br />
from keras.models import Model <br />
from keras.layers.core import Dense, Activation <br />
from keras.layers import SimpleRNN, Lambda, Input, Add, TimeDistributed, Concatenate, Dot <br />
import random <br />
from keras import backend as K <br />
from transformer import ScaledDotProductAttention, LayerNormalization <br />
import argparse <br />
import soundfile as sf <br />
from scipy import signal <br />
from transformers import AutoProcessor, AutoModelForPreTraining <br /> 
import warnings <br /> 
==========================================  <br />

For any new implementations using Rank-Net framework, we recommend to use this model code as base platform. All other three models are built on top of this training code.

For any questions please contact abinayreddy.naini@utdallas.edu, abinay.reddy01@gmail.com
