# Forecast ML

Main functionalities:
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
  - Neural Network training.
  - Creates gif from saved images.
  
  - Additional analysis tools.
    - Plot all time series inside a file.
    - Plot two time series in one graph for comparison.


------- changing how main_models work