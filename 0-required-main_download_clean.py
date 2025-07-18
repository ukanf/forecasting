from handle_data import download, clean
from handle_data.utils import utils
import os

__author__ = "Felipe Ukan Pereira"

@utils.timeit
def main():
    """
    Main function -
    """
    # FIPS code for state and counties
    # Code for parameters from usepa list
    G_STATES = [
        {
            'code': '06',
            'name': 'California',
            'counties': [
                {
                    'name': 'Los Angeles',
                    'code': '037',
                    'parameters': [
                        {
                            '44201': {
                                'name': 'Ozone',
                                'frequency': '1H',
                            }
                        },
                        # ] # REMOVE THIS
                        {
                            '42603': {
                                'name': 'NOX',
                                'frequency': '1H',
                            }
                        },
                        {
                            '62101': {
                                'name': 'Outdoor Temperature',
                                'frequency': '1H',
                            }
                        },
                        {
                            '88101': {
                                'name': 'PM2.5 - local conditions',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42401': {
                                'name': 'Sulfur Dioxide',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42101': {
                                'name': 'Carbon monoxide',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61101': {
                                'name': 'Wind speed - Scalar',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61102': {
                                'name': 'Wind direction - Scalar',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61103': {
                                'name': 'Wind speed - Resultant',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61104': {
                                'name': 'Wind direction - Resultant',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42602': {
                                'name': 'Nitrogen dioxide (NO2)',
                                'frequency': '1H',
                            }
                        },
                        {
                            '14129': {
                                'name': 'Lead (TSP) LC',
                                'frequency': '1H',
                            }
                        },
                        {
                            '63301': {
                                'name': 'Solar radiation',
                                'frequency': '1H',
                            }
                        }
                    ],
                },
                {
                    'name': 'Fresno',
                    'code': '019',
                    'parameters': [
                        {
                            '44201': {
                                'name': 'Ozone',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42401': {
                                'name': 'Sulfur Dioxide',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42603': {
                                'name': 'NOX',
                                'frequency': '1H',
                            }
                        },
                        {
                            '62101': {
                                'name': 'Outdoor Temperature',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42101': {
                                'name': 'Carbon monoxide',
                                'frequency': '1H',
                            }
                        },
                    ],
                },
                {
                    'name': 'Kern',
                    'code': '029',
                    'parameters': [
                        {
                            '44201': {
                                'name': 'Ozone',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42603': {
                                'name': 'NOX',
                                'frequency': '1H',
                            }
                        },
                        {
                            '62101': {
                                'name': 'Outdoor Temperature',
                                'frequency': '1H',
                            }
                        },
                        {
                            '88101': {
                                'name': 'PM2.5 - local conditions',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42401': {
                                'name': 'Sulfur Dioxide',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42101': {
                                'name': 'Carbon monoxide',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61101': {
                                'name': 'Wind speed - Scalar',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61102': {
                                'name': 'Wind direction - Scalar',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61103': {
                                'name': 'Wind speed - Resultant',
                                'frequency': '1H',
                            }
                        },
                        {
                            '61104': {
                                'name': 'Wind direction - Resultant',
                                'frequency': '1H',
                            }
                        },
                        {
                            '42602': {
                                'name': 'Nitrogen dioxide (NO2)',
                                'frequency': '1H',
                            }
                        },
                        {
                            '14129': {
                                'name': 'Lead (TSP) LC',
                                'frequency': '1H',
                            }
                        },
                        {
                            '63301': {
                                'name': 'Solar radiation',
                                'frequency': '1H',
                            }
                        },
                        {
                            '81102': {
                                'name': 'PM10 Total 0-10um STP',
                                'frequency': '1H',
                            }
                        },
                        {
                            '88502': {
                                'name': 'Acceptable PM2.5 AQI & Speciation Mass',
                                'frequency': '1H',
                            }
                        },
                        {
                            '62201': {
                                'name': 'Relative Humidity',
                                'frequency': '1H',
                            }
                        },
                        {
                            '63302': {
                                'name': 'Ultraviolet radiation',
                                'frequency': '1H',

                            }
                        },
                        {
                            '63303': {
                                'name': 'Infrared Radiation',
                                'frequency': '1H',
                            }
                        },
                        {
                            '64101': {
                                'name': 'Barometric pressure',
                                'frequency': '1H',
                            }
                        },
                        {
                            '65101': {
                                'name': 'Rain 24hr total',
                                'frequency': '1H',
                            }
                        },
                        {
                            '65102': {
                                'name': 'Rain/melt precipitation',
                                'frequency': '1H',
                            }
                        },
                        {
                            '66101': {
                                'name': 'Cloud cover',
                                'frequency': '1H',
                            }
                        },
                        {
                            '68105': {
                                'name': 'Average Ambient Temperature',
                                'frequency': '1H',
                            }
                        },
                        {
                            '68108': {
                                'name': 'Average Ambient Pressure',
                                'frequency': '1H',
                            }
                        },
                    ],
                },
            ],
        },
    ]
    raw_dataset_path = os.path.join('datasets', 'raw_datasets')
    cleaned_dataset_path = os.path.join('datasets', 'cleaned_datasets')

    ALL_PARAMETERS = {
        "D_states": G_STATES,
        "D_base_raw_url": 'https://aqs.epa.gov/api/rawData',
        "D_base_list_url": 'https://aqs.epa.gov/api/list',
        "D_prefix": raw_dataset_path,
        'C_prefix': cleaned_dataset_path,
        "D_format": "AQCSV",
        "D_user": "felipe.kpereira@live.com",
        "D_pass": "XXXXX",
        'C_states': G_STATES,
        'C_input_dataset_path': raw_dataset_path,
        'years': [x for x in range(1997, 2019)],
    }

    download.download_usepa_raw(ALL_PARAMETERS)

    # download.download_usepa_station_list(ALL_PARAMETERS)

    clean.clean_usepa(ALL_PARAMETERS)


if __name__ == "__main__":
    main()
