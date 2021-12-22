import csv
from torch.utils.tensorboard import SummaryWriter
import numpy as np

r1_steps = []
r1_values = []
r2_steps = []
r2_values = []

with open('prog.csv', newline='') as csvfile:
     tmp = csv.DictReader(csvfile)
     for row in tmp:
         r1_steps.append(row["Step"])
         r1_values.append(row["Value"])

with open('seq.csv', newline='') as csvfile:
     tmp = csv.DictReader(csvfile)
     for row in tmp:
         r2_steps.append(row["Step"])
         r2_values.append(row["Value"])

average = {}

smallest_sample = r1_values if len(r1_values) < len(r2_values) else r2_values

i = 0

average_values = []

for row in smallest_sample:
    average = (float(r1_values[i]) + float(r2_values[i])) / float(2)
    average_values.append(average)
    i += 1


#todo kør loop med average values.
# TensorBoard writer
writer = SummaryWriter("new_file") #husk navngiv

# Save logs to tensorboard
episode_idx = 200
i = 0
for row in smallest_sample:
    writer.add_scalar("evaluation/scores_min", 0, episode_idx)
    writer.add_scalar("evaluation/scores_max", 0, episode_idx)
    writer.add_scalar("evaluation/scores_mean", average_values[i], episode_idx) #Normalized score
    writer.add_scalar("evaluation/scores_std", 0, episode_idx)
    writer.add_histogram("evaluation/scores", 0, episode_idx)
    writer.add_scalar("evaluation/completions_min", 0, episode_idx)
    writer.add_scalar("evaluation/completions_max", 0, episode_idx)
    writer.add_scalar("evaluation/completions_mean", 0, episode_idx) #Completions
    writer.add_scalar("evaluation/completions_std", 0, episode_idx)
    writer.add_histogram("evaluation/completions", 0, episode_idx)
    writer.add_scalar("evaluation/nb_steps_min", 0, episode_idx)
    writer.add_scalar("evaluation/nb_steps_max",0, episode_idx)
    writer.add_scalar("evaluation/nb_steps_mean", 0, episode_idx)
    writer.add_scalar("evaluation/nb_steps_std", 0, episode_idx)
    writer.add_histogram("evaluation/nb_steps", 0, episode_idx)
    writer.add_scalar('evaluation/deadlocks_num', 0, episode_idx) #Deadlocks
    writer.add_scalar('evaluation/unfinished_agents_num', 0, episode_idx) #Total unfinished agents
    writer.add_scalar('evaluation/unfinished_not_deadlock', 0, episode_idx) #Unifnished - not deadlocked
    writer.add_scalar("evaluation/smoothed_score", 0, episode_idx) 
    writer.add_scalar("evaluation/smoothed_completion", 0, episode_idx) 
    writer.flush()
    print(episode_idx)
    episode_idx += 200
    i += 1


    #okay bare sæt de andre til 0.
    #og mean_socre til average.