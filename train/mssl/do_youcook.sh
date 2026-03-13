collection=Youcook2
visual_feature=i3d
clip_scale_w=0.5
frame_scale_w=0.5

root_path=./
device_ids=0
dataset_name=Youcook2
text_feat_path=../../depends/youcook2/youcook2_query_roberta.hdf5
caption_train_txt=../../depends/youcook2/training.txt
caption_test_txt=../../depends/youcook2/validation.txt
num_workers=12
# training
max_desc_len=120



local_hinge_weight=0.0
local_margin=0.2
global_soft_pos_weight=0.0
soft_pos_margin=0.1
soft_neg_margin=0.1



# training

query_or_caption=3

seed=2018


local_hinge_weight=0.9
global_soft_pos_weight=0.9
window_size=5

learing_rate=0.000085
caption_rate=1.0
query_or_caption=3
hard_negative_start_epoch=100

exp_id="baseline_lr${learing_rate}_hard_negative_start_epoch100_ours_all"
CUDA_VISIBLE_DEVICES=0 python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids $device_ids --dataset_name $dataset_name \
                    --text_feat_path $text_feat_path --caption_train_txt $caption_train_txt --caption_test_txt $caption_test_txt \
                    --caption_rate $caption_rate --max_desc_l $max_desc_len \
                    --local_margin $local_margin --local_hinge_weight $local_hinge_weight  \
                    --global_soft_pos_weight $global_soft_pos_weight --soft_pos_margin $soft_pos_margin \
                    --query_or_caption $query_or_caption --num_workers $num_workers \
                    --seed $seed --window_size $window_size --max_es_cnt 100 --lr $learing_rate --hard_negative_start_epoch $hard_negative_start_epoch   --global_caption