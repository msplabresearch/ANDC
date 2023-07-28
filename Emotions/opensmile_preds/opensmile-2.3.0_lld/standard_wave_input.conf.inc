///////////////////////////////////////////////////////////////////////////////////////
///////// > openSMILE configuration file for wave input <            //////////////////
/////////                                                            //////////////////
///////// (c) audEERING GmbH,                                        //////////////////
/////////     All rights reserverd.                                  //////////////////
///////////////////////////////////////////////////////////////////////////////////////



[componentInstances:cComponentManager]
instance[waveIn].type=cWaveSource

[waveIn:cWaveSource]
writer.dmLevel=wave
buffersize_sec = 5.0
filename=\cm[inputfile(I){test.wav}:name of input file]
start=\cm[startpoint(S){0}:startpoint in seconds]
end=\cm[endpoint(E){-1}:endpoint in seconds]
sampleRate=16000
monoMixdown=1
outFieldName = pcm
