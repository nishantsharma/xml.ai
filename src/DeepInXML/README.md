# pytorch-hier2hier

**[Documentation]**

This is a framework for XML-to-XML (hier2hier) models implemented in [PyTorch](http://pytorch.org).  The framework
has modularized and extensible components for hier2hier models, training and inference, checkpoints, etc.  This is an
alpha release. We appreciate any kind of feedback or contribution.

# Key Features 
1) Use of attention to find the appropriate character position and XML node to focus upon.
2) Use of GRU for encoding and decoding.  

# What's New in 0.0.1

* Compatible with PyTorch 0.4

# Roadmap
Hier2hier is an upcoming area.  The goal of this library is facilitating the development of such techniques and applications.  While constantly improving the quality of code and documentation, we will focus on the following items:

* Identification and evaluation with benchmarks;
* Provide more flexible model options, improving the usability of the library;
* Support features in the new versions of PyTorch.

# Installation

This package requires Python 3.6. We recommend creating a new virtual environment for this project (using virtualenv or conda).  

### Prerequisites

* Install python and ninja. Use following commands on MacOS for installation using macports,
  $ sudo port install python36
  $ sudo port install py36-pip
  $ sudo port select --set pip pip36
  $ sudo port select --set python python36
  $ sudo port install ninja 

* Install all python packages mentioned in requirements.txt. 
  $ sudo pip install -r requirements.txt


### Install from source

System level installation not yet supported.

# Get Started
### Prepare toy dataset
Run script to generate the reverse toy dataset.
The generated data is stored in data/toy_reverse by default
    ./scripts/generate.sh toy1
    ./scripts/generate.sh toy2

### Train
To continue last training run. 
    ./scripts/train.sh

To continue last training run for a specific domain. 
    ./scripts/train.sh --domain toy1
    ./scripts/train.sh --domain toy2

For help.
    ./scripts/train.sh -h

### Evaluate  
To evalaute latest trained domain. 
    ./scripts/evaluate.sh

To evaluate on domain toy1.
    ./scripts/evaluate.sh --domain toy1

For help.
    ./scripts/evaluate.sh -h

### Checkpoints
Checkpoints are organized by experiments and timestamps as shown in the following file structure

    experiment_dir
	+-- input_vocab
	+-- output_vocab
	+-- checkpoints
	|  +-- YYYY_mm_dd_HH_MM_SS
	   |  +-- decoder
	   |  +-- encoder
	   |  +-- model_checkpoint

The sample script by default saves checkpoints in the `experiment` folder of the root directory.  Look at the usages of the sample code for more options, including resuming and loading from checkpoints.

# Benchmarks

* WMT Machine Translation (Coming soon)

# Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/IBM/pytorch-seq2seq/issues/new) on Github.  For live discussions, please go to our [Gitter lobby](https://gitter.im/pytorch-seq2seq/Lobby).

We appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  

### Development Cycle
We are using 4-week release cycles, where during each cycle changes will be pushed to the `develop` branch and finally merge to the `master` branch at the end of each cycle.

### Development Environment
We setup the development environment using [Vagrant](https://www.vagrantup.com/).  Run `vagrant up` with our 'Vagrantfile' to get started.

The following tools are needed and installed in the development environment by default:
* Git
* Python
* Python packages: nose, mock, coverage, flake8

### Test
The quality and the maintainability of the project is ensured by comprehensive tests.  We encourage writing unit tests and integration tests when contributing new codes.

Locally please run `nosetests` in the package root directory to run unit tests.  We use TravisCI to require that a pull request has to pass all unit tests to be eligible to merge.  See [travis configuration](https://github.com/IBM/pytorch-seq2seq/blob/master/.travis.yml) for more information.

### Code Style
We follow [PEP8](https://www.python.org/dev/peps/pep-0008/) for code style.  Especially the style of docstrings is important to generate documentation.

* *Local*: Run the following commands in the package root directory
```
# Python syntax errors or undefined names
flake8 . --count --select=E901,E999,F821,F822,F823 --show-source --statistics
# Style checks
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```
* *Github*: We use [Codacy](https://www.codacy.com) to check styles on pull requests and branches.
