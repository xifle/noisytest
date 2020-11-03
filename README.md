# NoisyTest

NoisyTest is a command-line tool for noise-based system tests of (robotic) systems. It uses scalar noise (vibration) data from
a simulation environment or experiments to detect software flaws. The main objective behind this tool is the improvement
of virtual system tests to reduce the time required for real-world experiments. Especially in the robotics domain, 
real-world experiments are rather costly and too many flaws remain undetected in simulation.

NoisyTest is used for virtual system testing of the humanoid robot [Lola](https://www.mw.tum.de/en/am/research/current-projects/robotics/humanoid-robot-lola/).
In general, the tool may be used on any noise data to detect and characterize failures. 
Feel free to use NoisyTest for your purposes and let me know when you find it useful.

## Install

This is as easy as 
```pip install noisytest```.

## Usage

The install of NoisyTest registers an entry point `noisytest`, which works as a command line interface. It has a
built-in usage help:
```
$ noisytest --help
usage: noisytest [-h] [--pipeline NOISYTEST_PIPELINE]
                 [--config NOISYTEST_CONFIG] [-v]
                 {train,run} ...

This is NoisyTest 0.0.1

positional arguments:
  {train,run}
    train               Train a model from given data using default parameters
    run                 Run test on specified noise file

optional arguments:
  -h, --help            show this help message and exit
  --pipeline NOISYTEST_PIPELINE
                        the noisytest pipeline to use. (default:
                        default.noisy)
  --config NOISYTEST_CONFIG
                        noisytest config file name (default: noisytest-
                        config.json)
  -v, --verbosity       console output verbosity (default: None)
```

### Train a model

To train a model, we first need some noise-based training / validation data set. We use the dataset from the humanoid robot LOLA:

```bash
$ git clone https://github.com/am-lola/noisytest-data-lola.git data
```

We may then use noisytest to train a model:
```bash
$ noisytest --config data/noisytest-config.json train
```

The trained, self-contained pipeline (model + preprocessor) is written to disk after successful training: 'default.noisy'.
> :warning: NoisyTest uses [pickle](https://docs.python.org/3/library/pickle.html) for serialization. 
>Don't load pipelines of untrustworthy origin!

### Test noise data

To actually test a noise file for failures / flaws you run a test on a noise file:
```bash
$ noisytest run data/validation/earlycontacttoe.log
WARNING:possible oscillations in time region 0.0001-0.4
WARNING:possible oscillations in time region 6.7-7.1
...
```

### Data formats

Noise data is read from whitespace-separated files with two columns. One is the actual or simulated time in seconds, 
the other the scalar noise pressure estimate. Annotation data for the training and validation sets uses TOML files 
to mark individual time-frames.

## Disclaimer & Contributing

This project started as a mere proof of concept.  Although it already reached a state which can in general be used 
productively, it currently lacks some features --- and more important --- a solid test base. This will be fixed as soon
as I find time. In the meantime be warned of possible errors ;)

Feel free to contribute to this project.
