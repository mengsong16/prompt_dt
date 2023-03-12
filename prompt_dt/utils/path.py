import os

config_path_dict = {
    'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
    'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
    'ant_dir': "ant_dir/ant_dir_50.json",
    'ML1-pick-place-v2': "ML1-pick-place-v2/ML1_pick_place.json",
}

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(cur_path[:cur_path.find("/prompt-dt")], "prompt-dt")

task_config_path = os.path.join(root_path, 'task_config')
data_path = os.path.join(root_path, 'data')
runs_path = os.path.join(root_path, 'runs')
config_path = os.path.join(root_path, "configs")	
evaluation_path = os.path.join(root_path, "evaluation")
env_path = os.path.join(root_path, "envs")


if __name__ == "__main__": 
    print(cur_path)
    print(root_path)
    print(config_path)
    print(evaluation_path)
    print(task_config_path)
    print(data_path)
    print(runs_path)
    print(env_path)