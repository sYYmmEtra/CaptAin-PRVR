collection=youcook2
visual_feature=i3d
clip_scale_w=0.5
frame_scale_w=0.5

root_path=./test/method
device_ids=5
dataset_name=youcook2
text_feat_path=../../depends/generate_caption/youcook2/extracted_data/youcook2_query_roberta.hdf5
caption_train_txt=../../depends/generate_caption/youcook2/extracted_data/training.txt
caption_test_txt=../../depends/generate_caption/youcook2/extracted_data/validation.txt

# training
max_desc_len=120
caption_rate=1.0
global_soft_neg_weight=0.0
soft_neg_margin=0.1

global_margin=0.2
global_nce_weight=0.00
global_hinge_weight=0.0  



local_hinge_weight=0.4
local_margin=0.2
global_soft_pos_weight=0.3
soft_pos_margin=0.1

query_or_caption=3

plot_losses=video_grounding_loss
video_grounding_weight=0.5

local_caption_keep_num=1

num_workers=0
eval_context_bsz=500
eval_query_bsz=200

exp_id="yc2"
python train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids $device_ids --dataset_name $dataset_name \
                    --text_feat_path $text_feat_path --caption_train_txt $caption_train_txt --caption_test_txt $caption_test_txt\
                    --caption_rate $caption_rate --max_desc_l $max_desc_len \
                    --local_margin $local_margin --local_hinge_weight $local_hinge_weight \
                    --global_margin $global_margin --global_nce_weight $global_nce_weight \
                    --global_hinge_weight $global_hinge_weight --global_caption \
                    --global_soft_pos_weight $global_soft_pos_weight --soft_pos_margin $soft_pos_margin \
                    --global_soft_neg_weight $global_soft_neg_weight --soft_neg_margin $soft_neg_margin\
                    --query_or_caption $query_or_caption --video_grounding_weight $video_grounding_weight\
                    --plot_losses $plot_losses --local_caption_keep_num $local_caption_keep_num\
                    --num_workers $num_workers --eval_query_bsz $eval_query_bsz --eval_context_bsz $eval_context_bsz