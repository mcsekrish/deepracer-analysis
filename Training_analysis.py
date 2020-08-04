# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Imports
#
# Run the imports block below:

# +
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from deepracer.tracks import TrackIO, Track
from deepracer.tracks.track_utils import track_breakdown
from deepracer.logs import CloudWatchLogs as cw, \
    SimulationLogsIO as slio, \
    NewRewardUtils as nr, \
    AnalysisUtils as au, \
    PlottingUtils as pu, \
    ActionBreakdownUtils as abu

# Ignore deprecation warnings we have no power over
import warnings
warnings.filterwarnings('ignore')
# -


tu = TrackIO()
for f in tu.get_tracks():
    print(f)

# +
track: Track = tu.load_track("reInvent2019_track")

pu.plot_trackpoints(track)
# -

# AWS DeepRacer Console Logs
stream_name = 'rl_coach_1418_1593168185128' ## CHANGE This to your simulation application ID
fname = './logs/%s.log' %stream_name  # The log will be downloaded into the specified path
cw.download_log(fname, stream_prefix=stream_name)  # add force=True if you downloaded the file before but want to repeat

# +
EPISODES_PER_ITERATION = 20 #  Set to value of your hyperparameter in training

data = slio.load_data(fname)
df = slio.convert_to_pandas(data, episodes_per_iteration=EPISODES_PER_ITERATION)

df = df.sort_values(['episode', 'steps'])
# personally I think normalizing can mask too high rewards so I am commenting it out,
# but you might want it.
# slio.normalize_rewards(df)

#Uncomment the line of code below to evaluate a different reward function
#nr.new_reward(df, track.center_line, 'reward.reward_sample') #, verbose=True)
df.describe().T
# -
print('Last iteration number:',df['iteration'].max())
print('Total steps in last iteration:', df[df['iteration']==df['iteration'].max()]['steps'].max())

len(df)

# # Graphs of Training Progress

simulation_agg = au.simulation_agg(df)
au.analyze_training_progress(simulation_agg, title='Training progress')

# +
import numpy as np
fig, ax = plt.subplots(2, 2, figsize=(14, 12))
df_agg = simulation_agg[['iteration','progress','reward','throttle']]
if 1 in list(simulation_agg['complete']):
    df_agg['steps_if_complete'] = simulation_agg[simulation_agg['complete']==1]['steps']
else:
    df_agg['steps'] = simulation_agg['steps']
    
for c in range(len(df_agg.columns))[1:]:
    df_agg.boxplot(column = df_agg.columns[c], by='iteration', ax=ax.flatten()[c-1])
    ax.flatten()[c-1].set_xticks(range(0,df_agg['iteration'].max(),40))
    ax.flatten()[c-1].set_xticklabels(range(0,df_agg['iteration'].max(),40))
#     if c==2:
#         ax.flatten()[c].set_ylim(bottom = df_agg['throttle'].mean()-0.5, top = df_agg['throttle'].mean()+0.5)
# -


# ### Stats for all laps
au.scatter_aggregates(simulation_agg, 'Stats for all laps')
# ### Stats for complete laps
complete_ones = simulation_agg[simulation_agg['progress']==100]
au.analyze_training_progress(complete_ones, title='Complete laps progress')
# +
complete_ones = simulation_agg[simulation_agg['progress']==100]

if complete_ones.shape[0] > 0:
    au.scatter_aggregates(complete_ones, 'Stats for complete laps')
else:
    print('No complete laps yet.')
# -

import numpy as np
simulation_agg.groupby('iteration')['progress'].mean()

# ### Plot a training progress by quintiles


au.scatter_by_groups(simulation_agg, title='Quintiles')

au.scatter_by_groups(complete_ones, title='Quintiles')

#If you'd like some other colour criterion, you can add
#a value_field parameter and specify a different column
#pu.plot_track(df[df['iteration']>80], track)
df_scaled = df
df_scaled.loc[df_scaled['progress']==100,'reward']=0
# df_scaled=df_scaled[df_scaled['iteration']>100]
pu.plot_track(df_scaled, track)

# ## Data in tables

# View ten best rewarded episodes in the training
simulation_agg.nlargest(10, 'reward').T

# View five fastest complete laps
print('Total number of episodes ',max(simulation_agg['episode']))
complete_ones.nsmallest(10, 'time').T

# ### Path taken and throttle values in a particular episode

episode_id = simulation_agg[simulation_agg['progress']==100]['time'].idxmin()
print('Fastest episode number: ' "{:0.0f}".format(episode_id))
print('Lap time: ' "{:0.2f}".format(float(simulation_agg[simulation_agg['episode']==episode_id]['time'])))
print('Steps: ' "{:0.2f}".format(float(simulation_agg[simulation_agg['episode']==episode_id]['steps'])))
pu.plot_selected_laps(simulation_agg[simulation_agg['episode'] == episode_id], df, track)
pu.plot_grid_world(df[df['episode']==episode_id], track)

# ## Analyze the reward distribution for your reward function

# +
episode = df.groupby('closest_waypoint').mean()['reward']
episode[0:].plot.bar(x='closest_waypoint', y='reward')
fig = plt.gcf()
fig.set_size_inches(20,4)

# out = pd.cut(s, bins=[0, 0.35, 0.7, 1], include_lowest=True)
# ax = out.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(6,4))
# ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
# plt.show()
# -

# ### Path taken in a particular iteration

simulation_agg['iteration'].max()

iteration_id = simulation_agg['iteration'].max()-1
print('last complete iteration is ', iteration_id)
pu.plot_selected_laps([iteration_id], df, track, section_to_plot = 'iteration')
pu.plot_grid_world(df[df['iteration']==iteration_id], track)
iteration_id = round((simulation_agg['iteration'].max())/2)
print('in between iteration ', iteration_id)
pu.plot_selected_laps([iteration_id], df, track, section_to_plot = 'iteration')
pu.plot_grid_world(df[df['iteration']==iteration_id], track)
iteration_id = 1
print('first iteration ', iteration_id)
pu.plot_selected_laps([iteration_id], df, track, section_to_plot = 'iteration')
pu.plot_grid_world(df[df['iteration']==iteration_id], track)

