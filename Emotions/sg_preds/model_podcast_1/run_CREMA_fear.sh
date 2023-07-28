### Model type
## [wav2vec2-base, wav2vec2-large, wav2vec2-large-robust
## hubert-base, hubert-large, 
## wavlm-base, wavlm-base-plus, wavlm-large,
## data2vec-base, data2vec-large]
### Default model type
## wav2vec2: wav2vec2-large-robust
## hubert: hubert-large, 
## wavlm: wavlm-large,
## data2vec: data2vec-large


model_type=wav2vec2-large-robust

### Corpus
# USC-IEMOCAP 
# MSP-IMPROV
# MSP-PODCAST1.10
# MSP-PODCAST1.9
# CREMA-D
# NTHU-NNIME

corpus=CREMA-D
num_classes=ALL #four or ALL
output_num=1
label_rule=P       #P, M, D
partition_number=1
data_mode=primary #primary or secondary
seed=0
label_type=categorical
label_learning=hard-label




corpus_type=${corpus}_${num_classes}_${data_mode}

# Training
# python -u train.py \
# --device            cuda \
# --lr                1e-4 \
# --model_type        $model_type \
# --corpus_type       $corpus_type \
# --seed              $seed \
# --epochs            20 \
# --batch_size        32 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --output_num        $output_num \
# --label_type        $label_type \
# --label_learning    $label_learning \
# --corpus            $corpus \
# --num_classes       $num_classes \
# --label_rule        $label_rule \
# --partition_number  $partition_number \
# --data_mode         $data_mode \
# --model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${data_mode}/${label_rule}/partition${partition_number}/seed_${seed}

## Evaluation
#cpu or cuda



python -u test.py \
--root              $1 \
--device            cpu \
--model_type        $model_type \
--corpus_type       $corpus_type \
--seed              $seed \
--batch_size        1 \
--hidden_dim        1024 \
--num_layers        2 \
--output_num        $output_num \
--label_type        $label_type \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--label_rule        $label_rule \
--partition_number  $partition_number \
--data_mode         $data_mode \

--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${data_mode}/${label_rule}/partition${partition_number}/seed_${seed}
