python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.05 --width 4 --run_name "ct256_r18_n60_s005_w4" --gpu 4
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.05 --width 256 --run_name "ct256_r18_n60_s005_w256" --gpu 4
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.1 --width 4 --run_name "ct256_r18_n60_s010_w4" --gpu 4
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.1 --width 256 --run_name "ct256_r18_n60_s010_w256" --gpu 4
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.2 --width 4 --run_name "ct256_r18_n60_s020_w4" --gpu 4
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.2 --width 256 --run_name "ct256_r18_n60_s020_w256" --gpu 4
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.5 --width 4 --run_name "ct256_r18_n60_s050_w4" --gpu 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.5 --width 256 --run_name "ct256_r18_n60_s050_w256" --gpu 4 --dataparallel
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 1.0 --width 4 --run_name "ct256_r18_n60_s100_w4" --gpu 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 1.0 --width 256 --run_name "ct256_r18_n60_s100_w256" --gpu 4 --dataparallel