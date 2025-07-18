# Forecast ML

## Overview of files in this repo

Images:
  - automationML.png: overview of the approach taken for training the neural network models. This is the target state we were aiming for, everything is present in the repo except "Post-analysis ('brain')" which was not implemented.
  - general-view.png: these are the logical steps taken when dealing with the data and how we approached the research.
  - pre-processing.png: a breakdown of the steps/actions taken in the "pre-processing" step.

Python files:
  - 0-required-main_download_clean.py: it's a required step in the flow it downloads the data
  - 1-optional-main_cleaned_data_visualizations.py: a optional step to give us more insights into the data we just downloaded - this is good to visualize the correlation matrix and also the missing data (most of the times we have huge gaps of data which are hard to know we have by just looking at the csv file we downloaded).
  - 2-required-main_refine_all.py: a required step to prepare the dataset to be used for training a neural network.
  - 3-optional-main_generate_experiments_nn.py: optional step which will generate different permutations of the neural network we want to train.
  - 4-required-main_experiments_manager_nn.py: step that will train the neural networks - it will unpack the parameters provided, sets up a logger so if the job dies in a machine we will know what happened as it's logged in a file
  - 5-optional-gif_generator.py: generates a gif to visualize the outputted predictions
  - x-help-plot_all.py: a script to plot the parameters of a csv file. Used to visualize wind, temperature, PPM2.5, etc in graphs.
  - x-help-plot_same_graph.py: used to compare the predicted vs. measured values.
  - x-help-temporary_agent: this was used to simulate a idea I had at the time to have a "agent" that would orchestrate the experiments created in "3-optional-main_generate_experiments_nn.py".

folders:
  - extra: some other python files, mostly POCs of different types of models I was considering putting in the pipeline in step "4" which instead of being "nn" for neural network training it would be a generic training step for more than neural networks.
  - handle_data: scripts/functions used to download, clean, modify, prepare, "improve", etc the data to be trained.
  - models: scripts/functions used for the training of the models and while training them (logger, for example).


### Main functionalities

  - Download data from USEPA.
  - Parse and organize data in a time series.
  - Cleaned data visualization.
    - Generates missing data percentage view (per station and parameter).
    - Generates correlation matrix (for given station).
    - Generates Joint correlation plot (hex) for parameters in one station.
  - Refine data.
    - Fill missing data gaps.
        - Linear interpolation.
        - Average of samples for the same hour from the previous 3 days.
    - Treats outliers.
  - Generates json files with permutations of possible input parameters for a given neural network topology
  - Neural Network training.
  - Creates gif from saved images.
  
  - Additional analysis tools.
    - Plot all time series inside a file.
    - Plot two time series in one graph for comparison.


------- changing how main_models work