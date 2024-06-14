# master

## Requirements

### Cuda
This project can theoretically run on a CPU only, but this is not recommended. 
Installing Cuda is completely dependent on your hardware and OS, so the best source would be your favorite search engine.
We used https://www.cherryservers.com/blog/install-cuda-ubuntu up to and **not** including step 7. The Cuda Toolkit is not required.
Use this source at your own discretion.

### ILASP
This project requires the ILASP binary, which can be downloaded via the official ILASP [[1]](#1). website: https://www.ilasp.com/download
We used version 4.4.0. Note that only MacOS and Linux variants are supported.

Place this binary anywhere you wish, as long as you point to the right folder in the configuration.
We placed it in ./mario_phase1/asp/bin/

## Project Structure
### mario_phase0
This folder contains the pretrained Yolov8 object detection model and its paraphernalia. 
The model we use is located in /models/YOLOv8-Mario-lvl1-3/weights/best.pt
This model was trained on about 2000 labeled frames from levels 1-3 in world 1.
The file data.yaml contains the labels.
Both model and labels have to be properly referenced in the [Configuration](#configuration)

### mario_phase1
This folder contains the actual project, and is divided into the following parts:

#### root
* train_baseline: python script to train the baseline. The baseline implementation is a stripped and reworked version Montalvo et al's implementation [[2]](#2) (https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning)
* train.py: the python script to start training the neurosymbolic agent, which is the baseline augmented with neurosymbolic components.
* play.py: python script to play a trained model. To obtain these, train the baseline or the neurosymbolic agent.

Note that all three scripts use a [configuration](#configuration)

#### asp
Home to the predefined ASP programs for positioning, the ILASP mode biases and optional prior knowledge. There are several because the project can run in normal/pipeless mode and use constraints/no-constraints, and all four require a different set of ASP and ILASP mode bias.
 

#### callbacks
The architecture uses callbacks to provide logging, collect examples and perform induction.

#### ddqn
Home to the agents. These are reworks of Montalvo et al's implementation [[2]](#2). The agent running on constraints is separated for the author's convenience. It behaves slightly different when asking for advice.

#### experiments
Separate folder housing a script to run the timing experiments and the play experiments. The former uses a collection of different agent configurations, the latter needs one or more pretrained models in the designated folder.

#### mario_logging
This module is used for two purposes:
1) logging: the project uses an extensive logging mechanism to output timing and explainability logs
2) shared resources: some neurosymbolic components require input from others. Collected examples and induced advice is stored in and read from rolling logfiles. As a collateral, these also provide explainability.

The training, timing and play experiments all use their own log configuration.

#### symbolic components
Home to the neurosymbolic components. Their names are quite self-explanatory.

#### wrappers
The wrappers as provided by Montalvo et al's implementation [[2]](#2), with added wrappers for object detection and positioning.

<a name="configuration"></a>
## Configuration


## References

<a id="1">[1]</a>
M. Law, A. Russo, and K. Broda, ‘The ILASP system for learning Answer Set Programs’. 2015.

<a id="2">[2]</a>
J. Montalvo, Á. García-Martín, and J. Bescós, ‘Exploiting semantic segmentation to boost reinforcement learning in video game environments’, Multimedia Tools and Applications, vol. 82, no. 7, pp. 10961–10979, 2023.
