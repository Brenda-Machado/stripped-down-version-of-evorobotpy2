# evorobotpy2

## About
A stripped-down version of evorobotpy2 with OpenAI-ES-NE and some experiments results.

## Setup

First, if you are using a Debian-based Linux, install the following packages:
```bash
sudo apt-get install libgsl-dev libopenmpi-dev python3-virtualenv python3-dev python3-tk make g++
```
After that, create a virtual environment in the root directory:
```bash
virtualenv -p python3 venv
source venv/bin/activate
```
Then install the requirements:
```bash
pip install -r requirements.txt
```

To make things easier you can create an alias to run the simulator:
```
make create_alias
```

## Compiling

Before running the model, you need to compile the resources that you'll use.
The compile command is:
```
make compile_erdpole
make compile_evonet
```

## Running

To run a model, first go to the target environment, and then run the following command:
```bash
evrun -f {target environment}.ini
```
To see all execution options run:
```bash
evrun --help
```

## Contributing

If at some point you installed a new package, and therefore it will be needed to run the new code, you can update the requirements by running:
```bash
make update_requirements
```

## Concepts

A tool for training robots through evolutionary and reinforcement learning methods.

The tool is documented in [How to Train Robots through Evolutionary and Reinforcement Learning Methods](https://bacrobotics.com/Chapter13.html) which includes also a detailed tutorial with exercises.

For an introduction to the theory and to state-of-the-art research in this areas, see the associated open-access book [Behavioral and Cognitive Robotics: An Adaptive Perspective](https://bacrobotics.com).

## Credits

Please use this BibTeX to cite this repository in your publications:
```
@misc{evorobotpy2ne,
  author = {Bianchini, Arthur Holtrup and Nolfi, Stefano and Machado, Brenda Silva},
  title = {A stripped-down version of evorobotpy2 with OpenAI-ES-NE and some experiments results},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Brenda-Machado/stripped-down-version-of-evorobotpy2/}},
}
```
