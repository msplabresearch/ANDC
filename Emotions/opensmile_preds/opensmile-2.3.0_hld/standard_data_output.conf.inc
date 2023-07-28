///////////////////////////////////////////////////////////////////////////////////////
///////// > openSMILE configuration file for data output <           //////////////////
/////////                                                            //////////////////
///////// (c) audEERING GmbH,                                        //////////////////
/////////     All rights reserverd.                                  //////////////////
///////////////////////////////////////////////////////////////////////////////////////


/*
   This file can be included as data output file for standard feature
   extraction configuration files. It provides commandline options
   for the batch extraction GUI, and supports LLD and Functionals (summaries)
   saving.

   It requires the main extrator configuration file to provide the following
   data memory levels:  lld, lld_de, and func
 */

  //////////////////////////////////////////////////////////////////////
 ///////////////////  data output configuration  //////////////////////
//////////////////////////////////////////////////////////////////////

;;;;;;;;; output LLD features to ARFF ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

[componentInstances:cComponentManager]
;instance[arfflldsink].type=cArffSink
instance[arffsink].type=cArffSink


/*
[arfflldsink:cArffSink]
reader.dmLevel=lld;lld_de
frameIndex = 0
frameTime = 1
filename=\cm[output(O){?}:name of WEKA Arff output file, set to a valid filename to enable this output sink]
relation=\cm[relation{openSMILE_features}:arff relation attribute, feature set name and/or corpus name]
;instanceName=\cm[instname]
 ;; use this line instead of the above to always set the instance name to the
 ;; name of the input wave file
 ;instanceName=\cm[inputfile]
\{\cm[arfftargetsfile{arff_targets.conf.inc}:name of arff targets include file]}
append=\cm[appendstaticarff{0}:set to 0 to disable appending to an existing arff parameter summary file, given by the arffoutput option]
errorOnNoOutput = 1
*/



[arffsink:cArffSink]
reader.dmLevel=func
frameIndex = 0
frameTime = 0
filename=\cm[output(O){?}:name of WEKA Arff output file, set to a valid filename to enable this output sink]
relation=\cm[relation{openSMILE_features}:arff relation attribute, feature set name and/or corpus name]
;instanceName=\cm[instname]
 ;; use this line instead of the above to always set the instance name to the
 ;; name of the input wave file
 ;instanceName=\cm[inputfile]
\{\cm[arfftargetsfile{arff_targets.conf.inc}:name of arff targets include file]}
append=\cm[appendstaticarff{0}:set to 0 to disable appending to an existing arff parameter summary file, given by the arffoutput option]
errorOnNoOutput = 1


