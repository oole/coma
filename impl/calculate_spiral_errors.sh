calculate_error() {
echo ---------- Calculating error for $1 ----------
echo Calculating errors for CoMA and PCA
python errors.py --prediction results/$1_result.npy --data-dir data/$2
echo Finished
echo ----------------------------------------------
}


# Execute calculate_predictions.sh first.
# (split into two scripts, so that predictions and errors can be performed in different environments)
# Calculates the error for the given model predictions nd the specified data
# Extrapolation:
calculate_error spiral_lr8e3_bareteeth_coma bareteeth
calculate_error spiral_lr8e3_cheeks_in_coma cheeks_in
calculate_error spiral_lr8e3_eyebrow_coma eyebrow
calculate_error spiral_lr8e3_high_smile_coma high_smile
calculate_error spiral_lr8e3_lips_back_coma lips_back
calculate_error spiral_lr8e3_lips_up_coma lips_up
calculate_error spiral_lr8e3_mouth_down_coma mouth_down
calculate_error spiral_lr8e3_mouth_extreme_coma mouth_extreme
calculate_error spiral_lr8e3_mouth_middle_coma mouth_middle
calculate_error spiral_lr8e3_mouth_open_coma mouth_open
calculate_error spiral_lr8e3_mouth_side_coma mouth_side
calculate_error spiral_lr8e3_mouth_up_coma mouth_up

# Interpolation:
calculate_error spiral_lr8e3_sliced_coma sliced
