train_model() {
echo ---------- Training model for $1 ----------
echo Running Coma for $1 on $2 data
python main.py --coma-model-dir /abyss/home/tf-coma/coma-model --name $1 --data-dir /abyss/home/face-data/processed-data/$2 --mode train

echo ----------------------------------------------
}

# Trains the specified list of models, stores the checkpoints with the name in the specified coma-model-dir/checkpoint
# While training the tensorboard summaries are posted to the specified coma-model-dir/tensorboard as well.
train_model lr8e3_bareteeth_coma bareteeth
train_model lr8e3_cheeks_in_coma cheeks_in
train_model lr8e3_eyebrow_coma eyebrow
train_model lr8e3_high_smile_coma high_smile
train_model lr8e3_lips_back_coma lips_back
train_model lr8e3_lips_up_coma lips_up
train_model lr8e3_mouth_down_coma mouth_down
train_model lr8e3_mouth_extreme_coma mouth_extreme
train_model lr8e3_mouth_middle_coma mouth_middle
train_model lr8e3_mouth_open_coma mouth_open
train_model lr8e3_mouth_side_coma mouth_side
train_model lr8e3_mouth_up_coma mouth_up