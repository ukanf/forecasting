from handle_data import refine
from handle_data.utils import utils
import os

__author__ = "Felipe Ukan Pereira"
__copyright__ = "(c)  Felipe Ukan Pereira"
__license__ = "Proprietary"

@utils.timeit
def main():
    """
    Main function -
    """
    cleaned_dataset_path = os.path.join('datasets', 'cleaned_datasets')
    refined_dataset_path = os.path.join('datasets', 'refined_datasets')
    code_to_name_file = os.path.join('datasets', 'AQS_code_list', 'code_to_name.json')

    # ********* Arguments/options for refining data
    REFINE_PARAMETERS = {
        # FIPS code for state and counties
        # Code for parameters from usepa
        # o is for optional
        'R_country_code': '840',
        'R_state_code': '06',
        "R_prefix": refined_dataset_path,
        'R_input_dataset_path': cleaned_dataset_path,
        'R_code_to_name_file': code_to_name_file,
        # 'R_year_range': [x for x in range(1997, 2018)],
        'R_frequency': '1H',
        'R_max_missing_percentage': 20,
        'R_param_options': {
            '61101': {
            # this is WS Sca
            },
            '61102': {
            # this is WD Sca
                'R_o_upper_clip': {
                    # accepts only percentile
                    'percentile': 100,
                },
            },
            '61103': {
            # this is WS Res
            },
            '61104': {
            # this is WD Res
                'R_o_upper_clip': {
                    # accepts only percentile
                    'percentile': 100,
                },
            },
        }
    }

    # ********* Args for refining Kuwait data
    KUWAIT_REFINE_PARAMETERS = {
        'R_output_dataset_path': os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'refined'),
        'R_input_dataset_path': os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'cleaned'),
        # 'R_code_to_name_file': code_to_name_file,
        # 'R_frequency': '1H',
        'R_max_missing_percentage': 20,
        'R_min_nonzeros_in_day': 12,
        # 'R_param_options': {
        #
        #},

    }

    # ********* Args for refining US BRIAN data
    USBRIAN_REFINE_PARAMETERS = {
        'R_output_dataset_path': os.path.join('datasets', 'US_Brian', 'original', 'localformat', 'refined'),
        'R_input_dataset_path': os.path.join('datasets', 'US_Brian', 'original', 'localformat', 'cleaned'),
        # 'R_code_to_name_file': code_to_name_file,
        # 'R_frequency': '1H',
        'R_max_missing_percentage': 20,
        'R_min_nonzeros_in_day': 12,
        # 'R_param_options': {
        #
        # },
    }


    # refine.refine_usepa_dataset(REFINE_PARAMETERS)

    # refine.refine_kuwait_dataset(KUWAIT_REFINE_PARAMETERS)

    refine.refine_usbrian_dataset(USBRIAN_REFINE_PARAMETERS)

if __name__ == "__main__":
    main()
