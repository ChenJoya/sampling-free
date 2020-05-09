## Installation

### Requirements:
- GCC >= 4.9
- Anaconda (with python3)

### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name sampling-free
conda activate sampling-free

# this installs the right pip and dependencies for the fresh python
conda install ipython

# sampling-free and cocoapi dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python pillow==6.2.2

# install cocoapi
git clone https://github.com/cocodataset/cocoapi/  
cd cocoapi/PythonAPI
python setup.py build_ext install

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0 (10.0, 10.1, ... are also practicable)
conda install pytorch torchvision cudatoolkit=9.0

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install sampling-free
git clone https://github.com/ChenJoya/sampling-free.git
cd sampling-free

# the following will install the lib with symbolic links, 
# so that you can modify the files if you want and won't need to re-build it
python setup.py build develop
