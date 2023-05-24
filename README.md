# Acute Zero-Shot Imitation Learning with Task Prompting

<div align="center"> <img src="./assets/baseline-comparison.png" width=800> </div>

## Setup

Organize directories in this structure:

```
.
├── assets/
├── configs/
├── data/
│   ├── ant_dir/
│   ├── cheetah_dir/
│   ├── cheetah_vel/
│   ├── ...
├── envs/
│   ├── metaworld/
│   ├── mujoco-control-envs/
├── install_envs.sh
├── task_prompt_dt
├── README.md
├── requirements.txt
├── setup.py
```

To install requirements:

```setup
conda create --name task-prompt-dt python=3.10.10
conda activate task-prompt-dt
pip install -r requirements.txt
./install_envs.sh
```

- Our code was run on Ubuntu 22.04
- Note that mujoco-py depends on MuJoCo. Installation instructions can be found [here](https://github.com/openai/mujoco-py)

## Training + Evaluation

To train the model(s) in the paper, modify `main` in **task_prompt_dt/task_pdt_main.py** and run this command in **task_prompt_dt**:

```train
python task_pdt_main.py
```

To run evaluations, modify `main` in **task_prompt_dt/task_pdt_main.py** to include the
following, then run `python task_pdt_main.py`:

```eval
experiment_mix_env(config_filename="yaml_file_name_specified_in_configs_dir", mode="eval") # mode: ['train', 'eval', 'demo']
```

## Results
<div align="center"> <img src="./assets/results-table.png" width=800> </div>

## Acknowledgements
Our code is based primarily on [prompt-dt](https://github.com/mxu34/prompt-dt/)
