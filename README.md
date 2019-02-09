# Deep learning XML transoformations 


**[Documentation]**

This is a deep learning framework for automating XML-to-XML (hier2hier) transformations. It is implemented in
[PyTorch](http://pytorch.org).  The framework has modularized and extensible components for training, debugging,
inference and checkpoints etc.  This is an alpha release. We appreciate any kind of feedback or contribution.

# Key Features 

1) Hierarchical flow of information suited for XML, instead of linear flow in typical seq2seq models.
2) Use of attention to find the appropriate character position/XML node/node attribute to focus upon.
3) Use of GRU RNN for encoding and decoding.
4) Support for Beam decoding for better accuracy.
5) Custom GPU implementation of performance critical modules.
6) Tensorboard integration(over pytorch tensors).
7) Use of [shortcut connections](https://datascience.stackexchange.com/questions/22118/why-do-we-need-for-shortcut-connections-to-build-residual-networks) between layers in the network for a more stable convergence.
8) Use of memory networks for incorporating static content.
9) Use of pointer networks for better ability to verbatim copy of text data from input XML. 
10) Schema versioning: We keep tweaking our models. We often need a way to migrate training done on our old model into new schema.
   This can be called a kind of "self-transfer learning. This is supported via schema versioning.

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

# Get Started
### Prepare toy dataset
Run script to generate the reverse toy dataset.
The generated data is stored in data/inputs/<domainId> etc. by default
    ./scripts/generate.sh --domain toy1
    ./scripts/generate.sh --domain toy2

To get help on generation parameters, give the following command.
    ./scripts/generate.sh --domain toy1 --help
    ./scripts/generate.sh --domain toy2 --help

### Train
To continue last training run. 
    ./scripts/train.sh

To continue last training run for a specific domain. 
    ./scripts/train.sh --domain toy1
    ./scripts/train.sh --domain toy2

For help.
    ./scripts/train.sh -h

### Evaluate a model 
To evalaute latest trained model of a domain. 
    ./scripts/evaluate.sh --domain <domainId>

To evaluate on domain toy1.
    ./scripts/evaluate.sh --domain toy1

For help.
    ./scripts/evaluate.sh -h

### Checkpoints
Checkpoints are organized by domainId, runNo, modelSchemaNo and function as shown in the following file structure.

    data/
      +-- training/
            +-- runFolders/
                  +-- run.<runNo>.<domainId>_<modelSchemaNo>/
                  +-- run.00000.toy1_0/
                        +-- Chk<epochNo>.<batchNo>/
                              +-- input_vocab*.pt
                              +-- output_vocab.pt
                              +-- model.pt
                              +-- modelArgs
                              +-- trainer_states.pt
      +-- testing/
            +-- runFolders/
                  +-- run.00000.toy1_0/
                        +-- Chk*/
                  +-- run.<runNo>.<domainId>_<modelSchemaNo>/
                        +-- Chk*/
      +-- inputs/
            +-- <domainId>/
                  +-- dev/
                        +-- dataIn*.xml
                        +-- dataOut*.xml
                  +-- test/
                        +-- dataIn*.xml
                        +-- dataOut*.xml
                  +-- train/
                        +-- dataIn*.xml
                        +-- dataOut*.xml

The sample script by default saves checkpoints in the `inputs/<domainId>` folder of the root directory. Look
at the usages of the sample code for more options, including resuming and loading from checkpoints.

# Roadmap
The goal of this library is facilitating the development of XML-to-XML transformation techniques and applications.
Currently the greatest challenge is that XML files are quit big and seq2seq decoders take too much time training
overthem.

While constantly improving the performnce, quality of code and documentation, we will also focus on the following items:

* Identification and evaluation with benchmarks;
* Provide more flexible model options, improving the usability of the library;
* Support features in the new versions of PyTorch.

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
