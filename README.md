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
Additionally, you can download the models by running:
```
bash download_models.sh
```
Additionally we use "aligner" for the MFA environment as defined in [MFA installer](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)



### Executing program
* Create a directory and place all collected audio/video clips inside (i.e., /ANDC_batches/2023_7_29/audios)
* Run the script providing the batch directory
```
bash run.sh -r /ANDC_batches/2023_7_29
```
* All files will be saved inside the provided directory. Mainly Short_files.json will be created which aggregates all the information (segmentation, gender, music, SNR, emotions, etc). Furthermore, a directory will be created named Outputs which will contains the short and long audio clips as well as extracted features such as ASR.


<h2 align="left" id="coming-soon">TODO 🗓</h2>
* [x] Add details for the json files
* [x] Add details on the file structure used in Outputs
* [x] Remove extracted features once all inference has been complete
* [x] Add GPU support


# Contribute to the project
We highly encourage contribution to this repository. Feel free to fork it and add your own models and play with the code. We provide a template in /Template that can be used as the starting point. Once you develope your code, you can add the the run command and your own environment to the run.sh file. Finally, create a pull request if you want to contribute and share your models!

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
