# Moleculenet

Installation & Setup
```sh
git clone https://github.com/sean94011/Moleculenet.git

cd <path/to/setup.sh>

sh setup.sh
```

## Run Deepchem
```sh
cd deepchem

python ecoli.py
```

## Run Chemprop
* Directly Train
```sh
python train.py --data_path </path/to/Moleculenet/data/ecoli.csv> --dataset_type classification --save_dir ecoli_checkpoints
```
* Hyperparameter Optimization First and then Train
```sh
python hyperparameter_optimization.py --data_path </path/to/Moleculenet/data/ecoli.csv> --dataset_type classification --num_iters 100 --config_save_path config_ecoli

python train.py --data_path </path/to/Moleculenet/data/ecoli.csv> --dataset_type classification --config_path config_ecoli
```
