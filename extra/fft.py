import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Felipe Ukan Pereira"
__copyright__ = "(c)  Felipe Ukan Pereira"
__license__ = "Proprietary"

map_code_to_name = {
    "44201": "Ozone",
    "42603": "NOX",
    "62101": "Outdoor Temperature",
    "88101": "PM2.5 - local conditions",
    "42401": "Sulfur Dioxide",
    "42101": "Carbon monoxide",
    "61101": "Wind speed - Scalar",
    "61102": "Wind direction - Scalar",
    "61103": "Wind speed - Resultant",
    "61104": "Wind direction - Resultant",
    "42602": "Nitrogen dioxide (NO2)",
    "14129": "Lead (TSP) LC",
    "63301": "Solar radiation",
    "68108": "Average Ambient Pressure",
    "68105": "Average Ambient Temperature",
    "66101": "Cloud cover",
    "65101": "Rain 24hr total",
    "65102": "Rain/melt precipitation",
    "64101": "Barometric pressure",
    "63303": "Infrared Radiation",
    "63302": "Ultraviolet radiation",
    "62201": "Relative Humidity",
    "88502": "Acceptable PM2.5 AQI & Speciation Mass",
    "81102": "PM10 Total 0-10um STP",
}


def my_fft(values, column):
    """ Fast Fourier Transform
    by Felipe Ukan - 
    """
    fourier = np.fft.rfft(values-np.mean(values))
    freq = np.fft.rfftfreq(len(values), 0.001)

    fig, ax = plt.subplots()
    ax.plot(freq, abs(fourier.real))
    plt.title(column)
    plt.ylabel("Absolute Magnitude")
    plt.xlabel("Frequency [kHz]")
    plt.show()


if __name__ == "__main__":
    data_file_name = 'datasets/refined_datasets/1549049855/dataset.csv'
    df = pd.read_csv(data_file_name, header=0, index_col=0)

    # my_fft(df)
    for column in df.columns:
        my_fft(df[column].values, column)


