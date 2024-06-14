# Neurosymbolic Reinforcement Learning: Super Mario

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
Each training, timing or playing run needs to be configured. The configuration dictionary contains the following entries that need to be set in order for the agent to work properly:

* name: optional. Appears in the logs as handy identifier. For instance: P1_pipeless
* environment: The Mario environment to load
* interval_frequency: frequency with which the IntervalCallback is called. Note that its logger is switched off by default, as setting this interval to 1 can create tremendously large logfiles.
* checkpoint_frequency: frequency with which intermediate models are saved
* checkpoint_dir: folder where the intermediate models are saved
* display: Displays the visual game when True. Slows training, but is fun for playing.
* save_replay_buffer: save the replay buffer as well. Useful for continuing training on intermediate models
* detector_model_path: path to the Yolo model
* detector_label_path: path to the labels for the Yolo model
* positions_asp: path to the ASP to transform bounding boxes to positions
* show_asp: path to the ASP that reduces the atoms in the position Answer Set to only those that are useful for relative positioning
* relative_positions_asp: path to the ASP to transform positions to an object/relational representation. There are two, one for each normal/pipeless variant. 
* show_closest_obstacle_asp: path to the ASP that reduces the object/relational representation to only the closest obstacle and mario. This reduces overhead for applying advice. There are two, one for each normal/pipeless variant.
* advice_asp: path to prior knowledge. Useful for induction-less agents. For instance './asp/advice_constraints.lp'
* show_advice_asp: path to ASP program reducing the advice Answer Set to a clear action directive
* ilasp_binary: path to the ILASP binary
* ilasp_mode_bias: path to the ILASP mode bias. There are four, one for each constraints/no-constraints and normal/pipeless combination
* pipeless: False if normal, True if pipeless
* constraints: False if constraints, True is no-constraints
* bias: used for conflict resolution in examples.
* choose_intrusive: False for the non-instrusive algorithm, True for intrusive algorithm in the DDQN
* forget: True for memory retention, False to forget
* positive_examples_frequency: frequency with which positive examples are tried
* symbolic_learn_frequency: frequency with which induction takes place
* max_induced_programs: maximum number of induced programs. This is useful to stop induction when a certain threshold is reached

### Valid combinations
The following are valid constraints/no-constraints and normal/pipeless combinations. Using an invalid combination will most likely result in UNSATISFIABLE induction.

#### normal positioning - no constraints
Uses naive positioning and induces only positive rules. The pipeless parameter instructs the ExampleCollector to produce examples that fit the naive positioning and subsequent positive rule induction.

        "relative_positions_asp": './asp/relative_positions.lp',
        "show_closest_obstacle_asp": './asp/show_closest_obstacle.lp',
        "advice_asp": './asp/advice.lp', # or None
        "ilasp_mode_bias": './asp/ilasp_mode_bias_compact.las',
        "pipeless": False,
        "choose_intrusive": False, # recommended

#### pipeless positioning - no constraints
Uses discriminative positioning to exclude pipes and induces only positive rules. The pipeless parameter instructs the ExampleCollector to produce examples that fit the discriminative positioning and subsequent pipeless rule induction.

        "relative_positions_asp": './asp/relative_positions_ext.lp',
        "show_closest_obstacle_asp": './asp/show_closest_obstacle_ext.lp',
        "advice_asp": './asp/advise_ext.lp', # or None. Yes, typo in file name
        "ilasp_mode_bias": './asp/ilasp_mode_bias_compact_ext.las',
        "pipeless": True,
        "choose_intrusive": False, # recommended

#### normal positioning - constraints
Uses naive positioning and induces only constraints. The pipeless parameter instructs the ExampleCollector to produce examples that fit the naive positioning and subsequent constraint induction.


        "relative_positions_asp": './asp/relative_positions.lp',
        "show_closest_obstacle_asp": './asp/show_closest_obstacle.lp',
        "advice_asp": './asp/advice_constraints.lp', # or None
        "ilasp_mode_bias": './asp/ilasp_mode_bias_positive_constraints.las',
        "pipeless": False,
        "choose_intrusive": False, # recommended

#### pipeless positioning - constraints
Uses discriminative positioning to exclude pipes and induces only constraints. The pipeless parameter instructs the ExampleCollector to produce examples that fit the discriminative positioning and subsequent pipeless constraint induction.

        "relative_positions_asp": './asp/relative_positions_ext.lp',
        "show_closest_obstacle_asp": './asp/show_closest_obstacle_ext.lp',
        "advice_asp": './asp/advice_pipeless_constraints.lp', # or None
        "ilasp_mode_bias": './asp/ilasp_mode_bias_positive_constraints_ext.las',
        "pipeless": True,
        "choose_intrusive": False, # recommended


### mario_visualisation
Home to the Jupyter Notebooks. Note that these work on preprocessed .csv files (available upon request), which in turn are processed from the raw log files (very large, also available upon request).
Preprocessing is done by the preprocessing scripts in the same folder.

## References

<a id="1">[1]</a>
M. Law, A. Russo, and K. Broda, ‘The ILASP system for learning Answer Set Programs’. 2015.

<a id="2">[2]</a>
J. Montalvo, Á. García-Martín, and J. Bescós, ‘Exploiting semantic segmentation to boost reinforcement learning in video game environments’, Multimedia Tools and Applications, vol. 82, no. 7, pp. 10961–10979, 2023.
