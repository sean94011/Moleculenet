# install miniconda & deepchem-gpu 2.3

echo ' '
echo 'Installing Miniconda'
echo ' '

wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

sudo bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local

echo ' '
echo 'Installing Deepchem'
echo ' '

sudo conda install -y -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0

echo ' '
echo 'Downgrading scikit-learn to version 0.22'
echo ' '

pip instal scikit-learn==0.22

# install CUDA 9.2

echo ' '
echo 'Installing CUDA 9.2'
echo ' '

curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

sudo apt-get update

sudo apt-get install cuda-9-2

sudo nvidia-smi -pm 1

sudo nvidia-smi -ac 2505,875

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc

source ~/.bashrc

nvidia-smi

# Install pytorch GPU

echo ' '
echo 'Installing Pytorch GPU'
echo ' '

sudo conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

# install chemprop

echo ' '
echo 'Installing Chemprop'
echo ' '

git clone https://github.com/chemprop/chemprop.git

cd chemprop

sudo conda env create -f environment.yml

# Finish
echo ' '
echo 'Everything Installed Successfully!'
