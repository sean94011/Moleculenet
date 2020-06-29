# Y2 Lab AI Group: Moleculenet

## Installation & Setup
```sh
git clone https://github.com/sean94011/Moleculenet.git

sh Moleculenet/setup.sh
```

## Run Deepchem
```sh
python <path/to/Moleculenet/deepchem/ecoli.py>
```

## Run Chemprop
* Go to Chemprop and activate the environment
```sh
cd <path/to/chemprop>

source/conda activate chemprop
```
* Directly Train with Class Balancing
```sh
python train.py --data_path </path/to/Moleculenet/data/ecoli.csv> --dataset_type classification --save_dir ecoli_checkpoints --class_balance
```
* Hyperparameter Optimization First and then Train
```sh
python hyperparameter_optimization.py --data_path </path/to/Moleculenet/data/ecoli.csv> --dataset_type classification --num_iters 100 --config_save_path config_ecoli

python train.py --data_path </path/to/Moleculenet/data/ecoli.csv> --dataset_type classification --config_path config_ecoli
```
* Prediction on Library with Trained Model
```sh
python predict.py --test_path <path/to/library> --checkpoint_dir ecoli_checkpoints --preds_path ecoli_preds.csv
```
