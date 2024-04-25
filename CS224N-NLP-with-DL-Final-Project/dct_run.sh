# python3 simplebaseline_with_fft_only.py --use_gpu --option finetune --lr 5e-5 --weight_decay 0.01
# python3 simplebaseline_with_dct_only.py --use_gpu --option finetune --lr 5e-5 --weight_decay 0.01
# python3 simplebaseline_with_dct_with_filter_only.py --use_gpu --option finetune --lr 5e-5 --weight_decay 0.01

python3 simplebaseline.py --use_gpu --option finetune --lr 1e-5 --weight_decay 1e-7 --epochs 15
python3 simplebaseline_with_dct_only.py --use_gpu --option finetune --lr 1e-5 --weight_decay 1e-7 --epochs 15
python3 simplebaseline_with_fft_only.py --use_gpu --option finetune --lr 1e-5 --weight_decay 1e-7 --epochs 15