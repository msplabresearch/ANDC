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
Once all dependencies are installed, you can setup the different conda environments to be used by going to /env_setup and running 
```
bash create_conda_envs.sh
```
Additionally we use "aligner" for the MFA environment as defined in [MFA installer](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)

### Executing program
* Create a directory and place all collected audio/video clips inside (i.e., /ANDC/audios)
* Run the script proding the root directory containing the audios
```
bash run.sh -r /ANDC
```
* A directory will be created in the root directory named Outputs (i.e., /ANDC/Outputs) that contains all the generated files. Short_files.json is the json file which contains all the information needed to filter the short files (2.75-11 seconds) for annotations.

#TODO: add details for the json files
#TODO: add details on the file structure used in Outputs
#TODO: add Luz's code


#TODO: remove extracted features once all inference has been complete


# THE PIPELINE 
* 1st how to use the pipeline
* 2nd how to contribute with their model
#TODO: WRITE ABOUT HOW ANYBODY CAN SHARE THEIR MODELS HERE TO CONTRIBUTE IN THIS WORK WITHOUT DISTURBING THE FLOW OF THE PIPELINE.


## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
