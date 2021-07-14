from handle_data.utils import utils
from handle_data import data_visualization
import os

__author__ = "Felipe Ukan Pereira"
__copyright__ = "(c) 2019 Lakes Environmental Software Inc. Felipe Ukan Pereira"
__license__ = "Proprietary"

# todo add "default_params" dict with the default values

@utils.timeit
def main():
    """
    Main function -
    """
    cleaned_dataset_path = os.path.join('datasets', 'cleaned_datasets')
    refined_dataset_path = os.path.join('datasets', 'refined_datasets')
    code_to_name_file = os.path.join('datasets', 'AQS_code_list', 'code_to_name.json')
    code_to_name_min_file = os.path.join('datasets', 'AQS_code_list', 'code_to_name_min.json')
    code_to_parameter_min_file = os.path.join('datasets', 'AQS_code_list', 'code_to_parameter_min.json')

    # ********* Options for the visualization methods
    GET_VIS_PARAMETERS = {
        # FIPS code for state and counties
        # Code for parameters from usepa
        # o is for optional
        "R_prefix": refined_dataset_path,
        'R_input_dataset_path': cleaned_dataset_path,
        'R_code_to_name_file': code_to_name_file,
        'R_code_to_name_min_file': code_to_name_min_file,
        'R_code_to_parameter_min_path': code_to_parameter_min_file,
        # 'R_year_range': [x for x in range(1997, 2019)],
        # 'R_frequency': '1H',
        'R_o_filter_parameters': False,  # this is a filter.. use ['parametercode', 'parametercode']
        # visualization using missing percentage:start
        'R_o_visualize_missing_percentage': True,
        'R_o_visualize_missing_percentage_group_by': 'parameter',  # either station or parameter
        # visualization using missing percentage:end
        # visualization jointplot:start
        'R_o_visualize_jointplot_marginal': False,
        # visualization jointplot:end
        # visualization correlation matrix:start
        'R_o_visualize_correlation_matrix': False,
        'R_o_visualize_correlation_matrix_min_percent_todrop': 20,  #
        # visualization correlation matrix:end
    }
    if GET_VIS_PARAMETERS['R_o_visualize_missing_percentage']:
        data_visualization.vis_missing_percentage(GET_VIS_PARAMETERS)
    if GET_VIS_PARAMETERS['R_o_visualize_jointplot_marginal']:
        data_visualization.vis_jointplot_marginal(GET_VIS_PARAMETERS)
    if GET_VIS_PARAMETERS['R_o_visualize_correlation_matrix']:
        data_visualization.vis_correlation_matrix(GET_VIS_PARAMETERS)



if __name__ == "__main__":
    main()
