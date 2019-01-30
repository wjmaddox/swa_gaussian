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
* ```<METHOD>``` &mdash; method to evaluate 
      - `SWAG`
      - `KFACLaplace`
      - `SGD`
      - `Dropout`
      - `SWAGDrop`
* ```<SCALE>``` &mdash; scale parameter for re-scaling the posterior approximation; in the experiments we set it equal to `0.5` for `SWAG` and to `1.` for `SWAG-diagonal` and `KFAC-Laplace` (default: `1`)
* ```<SAMPLES>``` &mdash; number of samples from the approximate posterior to use in Bayesian model averaging (default: `30`)
