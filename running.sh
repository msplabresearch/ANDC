#!/bin/bash
############################################################
# Help: Arguments and help command                         #
############################################################

Help()
{
   # Display Help
   echo "ANDC Pipeline v0.1"
   echo
   echo "Syntax: $0 -r root_dir"
   echo
}

while getopts 'r:' opt; do
        case $opt in
                r) root_dir=${OPTARG} ;;
                *)
                        echo 'Error in command line parsing' >&2
                        exit 1
        esac
done
eval "$(conda shell.bash hook)"

opensmile_path="/home/a/Desktop/MSP-Podcast/pipeline/emotion_retrieval_final/opensmile/opensmile/build/progsrc/smilextract/SMILExtract";


############################################################
# Step1: segment the audio clips (Whisper)                 #
############################################################


python ./Segmentation/Whisper/segmentation.py --root $root_dir

conda activate aligner
    cd Segmentation/MFA
        bash step2_MFA_run.sh -r "$root_dir"
    cd -
conda deactivate

############################################################
# Step2: Sentiment pred                                    #
############################################################

python ./Sentiment/prediction_WordSentiment.py --root $root_dir


############################################################
# Step3: Gender pred                                       #
############################################################
conda activate winston_base 
    python ./Gender/gender_prediction.py --root $root_dir
conda deactivate


############################################################
# Step4: Gender pred                                       #
############################################################

conda activate winston_base 
    python ./Music/music_predict.py --root $root_dir
    python ./Music/music_filter.py --root $root_dir
conda deactivate


############################################################
# Step5: Extract opensmile features                        #
############################################################

conda activate winston_base 
    python ./Emotions/opensmile_preds/extract_features_llds_hlds.py --root $root_dir --opensmile $opensmile_path
    python ./Emotions/opensmile_preds/prediction_AttenVec.py --root $root_dir --dataset MSP-Podcast
    python ./Emotions/opensmile_preds/prediction_AttenVec.py --root $root_dir --dataset MSP-IMPROV 
    python ./Emotions/opensmile_preds/prediction_AttenVec.py --root $root_dir --dataset IEMOCAP 
    python ./Emotions/opensmile_preds/prediction_LadderNet.py --root $root_dir
conda deactivate



############################################################
# Step6: Extract opensmile features                        #
############################################################


conda activate HF
    cd ./Emotions/sg_preds/model_podcast_1/
    bash ./run_CREMA_fear.sh $root_dir
    cd -

    cd ./Emotions/sg_preds/model_podcast_2/
    bash run_CREMA_disgust.sh $root_dir
    cd -

    cd ./Emotions/sg_preds/model_podcast_3/
    bash run_podcast10_fear.sh $root_dir
    cd -
conda deactivate


############################################################
# Step7: Extract opensmile features                        #
############################################################
conda activate HF
    cd ./Emotions/rank/
    python w2v_feature_extraction.py --root $root_dir
    python ./ranking/test_feature_extraction.py --root $root_dir
    cd -
conda deactivate


############################################################
# Step8: Predict primary using opensmile feats             #
############################################################

conda activate ssl #

    cd Emotions/ssl
    python prediction_classifier.py --root $root_dir
    cd -
conda deactivate


############################################################
# Step9: ranknet VAD preds             #
############################################################

conda activate tf2_new
    cd Emotions/rank
    python prediction_ranknet_arousal.py --root $root_dir
    cd -
conda deactivate


############################################################
# Step9: ranknet VAD preds             #
############################################################

conda activate tf2_new
    cd Emotions/rank/ranking
    python testing_rank_valence.py --root $root_dir
    python testing_rank_dominance.py --root $root_dir
    python testing_rank_activation.py --root $root_dir
    cd -
conda deactivate



############################################################
# Step9: ranknet VAD preds             #
############################################################

conda activate tf2_new
    cd Emotions/preference_Ladder
    python testing_ladderpref_activation.py --root $root_dir
    python testing_ladderpref_dominance.py --root $root_dir
    python testing_ladderpref_valence.py --root $root_dir
    cd ..
conda deactivate