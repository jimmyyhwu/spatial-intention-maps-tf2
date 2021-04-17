# spatial-intention-maps-tf2

Spatial intention maps implemented with [TensorFlow 2](https://www.tensorflow.org) and [TF-Agents](https://www.tensorflow.org/agents). See [here](https://github.com/jimmyyhwu/spatial-intention-maps) for the original PyTorch implementation.

## Installation

```bash
# Create and activate new conda env
conda create -y -n my-conda-env python=3.7.10
conda activate my-conda-env

# Install mkl numpy
conda install -y numpy==1.19.2

# Install tensorflow-gpu
conda install -y tensorflow-gpu=2.4.1

# Install pip requirements
pip install -r requirements.txt

# Install shortest paths module (used in simulation environment)
cd shortest_paths
python setup.py build_ext --inplace
```
