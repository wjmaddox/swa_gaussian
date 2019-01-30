## Uncertainty experiments README

The script `experiments/uncertainty/uncertainty.py` evaluates the predictions of the models trained with `experiments/train/run_swag.py` on test data:
```
python experiments/uncertainty/uncertainty.py \
      --file=<FILE> \
      --save_path=<SAVEPATH> \
      --dataset=<DATASET> \
      --data_path=<PATH> \
      --model=<MODEL> \
      --method=<METHOD> \
      --scale=<SCALE> \
      --N=<SAMPLES> \
      [--cov_mat \]
      [--use_test \]
      [--use_diag \]
      [--split_classes=<SPLIT> \]
```
Parameters:
* ```<FILE>``` &mdash; path to the checkpoint
* ```<SAVEPATH>``` &mdash; path to save the predictions of the model
* ```<>```


### Image Classification experiments

Scripts to run results:
