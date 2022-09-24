python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.05 --width 8 --run_name "ct256_r18_n60_s005_w8" --gpu 5
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.05 --width 192 --run_name "ct256_r18_n60_s005_w192" --gpu 5
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.1 --width 8 --run_name "ct256_r18_n60_s010_w8" --gpu 5
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.1 --width 192 --run_name "ct256_r18_n60_s010_w192" --gpu 5
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.2 --width 8 --run_name "ct256_r18_n60_s020_w8" --gpu 5
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.2 --width 192 --run_name "ct256_r18_n60_s020_w192" --gpu 5
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.5 --width 8 --run_name "ct256_r18_n60_s050_w8" --gpu 5
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 0.5 --width 192 --run_name "ct256_r18_n60_s050_w192" --gpu 5 --dataparallel
python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 1.0 --width 8 --run_name "ct256_r18_n60_s100_w8" --gpu 5
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --config exps/caltech_r18.yaml --mislabel_ratio 0.4 --subset --subset_size 1.0 --width 192 --run_name "ct256_r18_n60_s100_w192" --gpu 5 --dataparallel