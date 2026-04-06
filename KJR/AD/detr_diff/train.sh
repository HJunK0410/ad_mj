nohup bash train_with_continuity6.sh
nohup bash train_with_continuity7.sh
nohup bash train_with_continuity8.sh
nohup bash train_with_continuity9.sh

nohup bash test_best.sh ./outputs/detr_diff_75
nohup bash test_best.sh ./outputs/detr_diff_50
nohup bash test_best.sh ./outputs/detr_diff_25
nohup bash test_best.sh ./outputs/detr_diff_10