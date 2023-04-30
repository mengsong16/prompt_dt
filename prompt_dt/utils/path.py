import os

config_path_dict = {
    'cheetah_vel': "cheetah_vel/cheetah_vel-40.json",
    'cheetah_dir': "cheetah_dir/cheetah_dir-2.json",
    'ant_dir': "ant_dir/ant_dir-50.json",
    "walker_param": "walker_param/walker_param-50.json",
    'ML1-pick-place-v2': "ML1-pick-place-v2/ML1-pick-place-v2-50.json",
    'ML1-reach-v2': "ML1-reach-v2/ML1-reach-v2-50.json",
    'ML1-sweep-v2': "ML1-sweep-v2/ML1-sweep-v2-50.json",
}

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(cur_path[:cur_path.find("/prompt-dt")], "prompt-dt")

task_config_path = os.path.join(root_path, 'task_config')
data_path = os.path.join(root_path, 'data')
runs_path = os.path.join(root_path, 'runs')
config_path = os.path.join(root_path, "configs")	
evaluation_path = os.path.join(root_path, "evaluation")
env_path = os.path.join(root_path, "envs")
demo_path = os.path.join(root_path, 'demo')
figure_path = os.path.join(root_path, "figures")

mnist_figure_path = os.path.join(figure_path, "mnist")
mnist_runs_path = os.path.join(runs_path, "mnist")

# code path
cvae_path = os.path.join(root_path, "prompt_dt", "cvae")
mnist_path = os.path.join(cvae_path, "mnist")



if __name__ == "__main__": 
    print(cur_path)
    print(root_path)
    print(config_path)
    print(evaluation_path)
    print(task_config_path)
    print(data_path)
    print(runs_path)
    print(env_path)