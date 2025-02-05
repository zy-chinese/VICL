## Exploring Task-Level Optimal Prompts for Visual In-Context Learning

We follow previous works to construct our code project. 

To test our methods proposed in this paper, you can use the bash bellow:

```bash
for seed in 0 1 2
    do
        for split in 0 1 2 3
            do
                CUDA_VISIBLE_DEVICES=5 python evaluate_segmentation.py \
                    --model mae_vit_large_patch16 \
                    --ckpt '/data1/xiangmu/models/checkpoint-3400.pth' \
                    --choose_num 6 \
                    --shots 16 \
                    --seed $seed \
                    --split $split \
                    --dataset_type pascal \
                    --base_dir /data1/dataset \
                    --output_dir  "../npys/output_dir_trn_16_s{$split}_seed{$seed}/" \
                    --method sum \
                    --trn 
            done
    done
```

`choose_num` is the max number of demonstration set.

`shots` is the number of training set.

`trn=True` means to get the results of validation set and test set when prompting by all possible demonstration set.

`sim=True` means to get the results of UnsupPR.

`aug=True` means to get the results by using different arrangements to create 8 new fused images. When `sim=True` , it also means to get the results of Prompt-SelF.



After getting the results of bash above, you can use `draw.py` to get the final results of top-K and greedy methods.

```
python draw.py
```

