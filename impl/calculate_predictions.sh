calculate_predictions() {
echo ---------- Calculating error for $1 ----------
echo Running Coma for $1 on $2 data
python main.py --coma-model-dir /abyss/home/tf-coma/coma-model --name $1 --data-dir /abyss/home/face-data/processed-data/$2 --mode test
echo Predictions for $1 on $2 saved
echo ----------------------------------------------
}

# Calculates the prediction using the specified model (-checkpoint) and the specified data
# Extrapolation:
calculate_predictions lr8e3_bareteeth_coma bareteeth
calculate_predictions lr8e3_cheeks_in_coma cheeks_in
calculate_predictions lr8e3_eyebrow_coma eyebrow
calculate_predictions lr8e3_high_smile_coma high_smile
calculate_predictions lr8e3_lips_back_coma lips_back
calculate_predictions lr8e3_lips_up_coma lips_up
calculate_predictions lr8e3_mouth_down_coma mouth_down
calculate_predictions lr8e3_mouth_extreme_coma mouth_extreme
calculate_predictions lr8e3_mouth_middle_coma mouth_middle
calculate_predictions lr8e3_mouth_open_coma mouth_open
calculate_predictions lr8e3_mouth_side_coma mouth_side
calculate_predictions lr8e3_mouth_up_coma mouth_up

# Interpolation:
calculate_predictions lr8e3_sliced_coma sliced