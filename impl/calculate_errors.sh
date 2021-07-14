calculate_error() {
echo ---------- Calculating error for $1 ----------
echo Running Coma for $1 on $2 data
python main.py --coma-model-dir /home/oole/coma-model/ccu2 --name $1 --data-dir /media/oole/Storage/Msc/processed-data/$2 --mode test
echo Predictions for $1 on $2 saved
echo Calculating errors for CoMA and PCA
python errors.py --prediction results/$1_result.npy --data-dir /media/oole/Storage/Msc/processed-data/$2
echo Finished
echo ----------------------------------------------
}

calculate_error lr8e3_bareteeth_bs16_210713_1256 bareteeth
# calculate_error lr8e3_cheeks_in_bs16_210713_1256 cheeks_in
# calculate_error lr8e3_eyebrow_bs16_210713_1440 eyebrow
# calculate_error lr8e3_high_smile_bs16_210713_1455 high_smile
# calculate_error lr8e3_lips_back_bs16_210713_1710 lips_back
# calculate_error lr8e3_lips_up_bs16_210713_1712 lips_up
# calculate_error lr8e3_mouth_down_bs16_210713_1911 mouth_down
# calculate_error lr8e3_mouth_extreme_bs16_210713_2001 mouth_extreme
# calculate_error lr8e3_mouth_middle_bs16_210713_2356 mouth_middle
# calculate_error lr8e3_mouth_open_bs16_210713_2356 mouth_open
# calculate_error lr8e3_mouth_side_bs16_210713_2356 mouth_side
# calculate_erro lr8e3_mouth_up_bs16_210713_2356 mouth_up