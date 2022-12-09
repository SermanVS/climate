## This is a set of neural networks for identifying and predicting climate events

#### Purpose

The source code in this repository attempts to solve a task of recognizing and predicting cyclone events using several metrics.

The recognition means to identify whether there is a cyclone event present on a current tick, while the prediction task means to identify whether there will be a cyclone event present ***w*** ticks from the current tick.

All networks are based on the formula
$$
\sigma (\sum(\sigma((w * x - a) * b)) - c).
$$
But most networks alter this formula slightly to reduce a number of parameters or to limit their range. Also, most networks perform maxpooling before implementing the formula above.

#### Data

The networks implemented here require specific metrics to learn properly. These metrics are ***MSLP_preproc***, ***LCC_w***, ***EVC_w***, ***degree_w***, ***closeness_w***. Each metric is a NumPy array of shape (36, 69, 113960) - 113960 ticks of images of size (36, 69).

#### Repository description

1. ***src*** - contains Jupyter Notebook files with implemented neural networks and necessary .py files with functions;
2. ***pretrained_models*** - contains state dictionaries of pretrained models, and also models themselves;
3. ***shuffle_cyclone.csv*** - contains shuffled ticks on which cyclone events were present;
4. ***shuffle_no_cyclone.csv*** - contains shuffled ticks on which cyclone events weren't present.

#### Prerequisites

To run the code in Jupyter Notebooks makes sure you have the following modules installed:

1. **NumPy**
2. **PyTorch**
3. **Pandas**
4. **Matplotlib**
5. **scikit-learn**
6. **scipy**
7. **tqdm_notebook**
8. **pathlib**
9. **configparser**
10. **Seaborn**

The file structure should be as follows. It is presumed that whoever attempts to run this code already has the ***data*** folder.

```
climate
|	README.md
|	shuffle_cyclone.csv
|	shuffle_no_cyclone.csv
|
└---src
|	|	Cyclone.ipynb
|	|	Cyclone_bs24.ipynb
|   |   Cyclone_bs24_de.ipynb
	...
|   |   train.py	
|
└---pretrained_models
|   |
|   └---models
|       |   ...
|   |
|   └---statedicts
|       |   ...
|    
└---data
|   |   cyclone_times.csv
|   |   tropical_cyclones_data_1982_2020.csv
|   |
|   └---ERA5
|       |
|       └---ERA5_MSL_1982_2020_3h_0.75
|           |   cyclones_events.npz
|           |   lat.txt
|           |   lon.txt
|           |   msl_1982_2001.nc
|           |   resulting_cube.npz
|           |   resulting_cube_land_masked.npz
|           |   resulting_cube_land_masked_and_preproc.npz
|           |   times.txt
|           |   tropical_cyclones_data_1982_2020.csv
|           |
|           └---_cyclones_5d
|           	|    ...
|           |
|           └---metrics_corr_land_masked_and_preproc_window_2d_delay_0d
|               |   metric_names.npy
|               |
|               └---diff_metrics
|               	|   ...
|               |
|				└---input_data
|               	|   ...
|               |
|               └---lgm_deviation_for_cyclones
|               	|   ...
|               |
|               └---network_metrics
|               	|   ...
|               |
|               └---probability_for_metrics
|                   |
|                   └---diff_metrics
|                   	|   ...
|                   |
|                   └---input_data
|                       |   MSLP_preproc.npy
|                       |	...
|                   |
|                   └---network_metrics
|                       |   closeness_w.npy
|                       |   degree_w.npy
|                       |   EVC_w.npy
|                       |   LCC_w.npy
|                       |   ...
```

#### Configure

The config file has the following location: **src/config.cfg**

There are 4 field that need to be set before running the code:

1. ***metric***. The following metrics are supported: **MSLP_preproc**, **LCC_w**, **EVC_w**, **degree_w**, **closeness_w**.

2. ***metric_path***. Path to metric data. The table below demonstrates the correlation between a metric and its path.

   | metric       | path                        |
   | ------------ | --------------------------- |
   | MSLP_preproc | input_data/MSLP_preproc     |
   | LCC_w        | network_metrics/LCC_W       |
   | EVC_w        | network_metrics/EVC_w       |
   | degree_w     | network_metrics/degree_w    |
   | closeness_w  | network_metrics/closeness_w |

3. ***mode***. There are two working modes: **recognize** and **predict**. In the former mode the network tries to recognize if there is a cyclone event present on a current tick. In the latter mode the network tries to predict whether there will be a cyclone event present ***w*** ticks from the current tick.
4. **w**. Defines number of ticks between the current tick and the tick used for prediction during the **predict** mode. Isn't used in the **recognized** more.

#### How to run

After setting the desired configuration, run the **run_all.py** script by typing the following command:

```bash
python run_all.py
```

This will run all Jupyter Notebooks in the **src** folder. 

**DO NOT CHANGE CONFIGURATION WHILE RUNNING** as each notebook tries to access the current version of the **config.cfg** file.

If you want to run only one notebook you can use VS Code and do it manually, or open the **run_one.py** file, edit the **notebook** variable and set it to the name of the notebook you want to run, and then run the command

```bash
python run_one.py
```

Both commands rewrite the original notebooks with the resulting notebooks. If that behavior isn't desired, the **run_all.py** and **run_one.py** scripts can be changed to save new notebooks under new names in separate locations.

