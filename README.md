# ANDC
Affective Naturalistic Database Consortium


## Getting Started

### Dependencies

Please make sure the following dependencies are installed before using this repository:

* openSMILE: https://www.audeering.com/research/opensmile/
* Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
* Whisper: https://github.com/openai/whisper
* Conda: https://docs.conda.io/en/latest/

### Installing
Once all dependencies are installed, you can setup the different conda environments to be used by going to /ANDC/env_setup and running 
```
bash create_conda_envs.sh
```
Additionally we use "aligner" for the MFA environment as defined in [MFA installer](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)

Finally, download the models used in this code from the following this [link](https://utdallas.box.com/s/rky9djcu03n7m9uvxqm47q5crurefokl)

### Executing program
* Create a directory and place all collected audio/video clips inside (i.e., /ANDC_batches/2023_7_29/audios)
* Run the script providing the batch directory
```
bash run.sh -r /ANDC_batches/2023_7_29
```
* All files will be saved inside the provided directory. Mainly Short_files.json will be created which aggregates all the information (segmentation, gender, music, SNR, emotions, etc). Furthermore, a directory will be created named Outputs which will contains the short and long audio clips as well as extracted features such as ASR.

#TODO: add details for the json files
#TODO: add details on the file structure used in Outputs
#TODO: add Luz's code


#TODO: remove extracted features once all inference has been complete


# THE PIPELINE 
* 1st how to use the pipeline
* 2nd how to contribute with their model
#TODO: WRITE ABOUT HOW ANYBODY CAN SHARE THEIR MODELS HERE TO CONTRIBUTE IN THIS WORK WITHOUT DISTURBING THE FLOW OF THE PIPELINE.

# Contribute to the project
We highly encourage contribution to this repository. Feel free to fork it and add your own models and play with the code. We provide a template in /Template that can be used as the starting point. Once you develope your code, you can add the the run command and your own environment to the run.sh file. Finally, create a pull request if you want to contribute and share your models!

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
