import os
import io
import requests
import pandas as pd
from .utils import utils

map_read_frequency_to_duration = {
    '1H': '1',
    'H': '1',
    '24H': 'X',
    # 24h: X is for average.. 7 is for observed and V for combined
    # 24h can also be X or V .. but they are calculated
    '12H': '6',
    '8H': '5',
    '6H': '4',
    '4H': '3',
    # '168H': 'A'
}


@utils.timeit
def download_usepa_station_list(download_params):
    """
    :param download_params:
    :return:
    """
    r = None

    for state in download_params['D_states']:
        curr_state = state['code']
        for county in state['counties']:
            curr_county = county['code']
            curr_download_url = download_params[
                                    'D_base_list_url'] + "?name=site&null&state={state}&county={county}".format(
                state=curr_state, county=curr_county)
            try:
                r = requests.get(curr_download_url, stream=True)
                if r.status_code == 200:
                    # print(r.content.decode(r.encoding))
                    result_dict = {}
                    for item in r.content.decode(r.encoding).split('\n'):
                        try:
                            key, value = item.split('	')
                        except ValueError:
                            key, value = item, ''
                        if key:
                            result_dict[key] = {
                                'name': value,
                                'has_data': utils.station_has_data(download_params, curr_state, curr_county, key),
                            }

                    full_out_file_path = os.path.join(download_params['C_prefix'], curr_state, curr_county,
                                                      'stations_list.json')
                    utils.write_json_to_file(result_dict, full_out_file_path)

                    if r:
                        r.close()
                else:
                    print("Error downloading file: ")
                    exit()
            except Exception as e:
                print('Error downloading file: {}'.format(e))
                if r:
                    r.close()
            continue
    return


@utils.timeit
def download_usepa_raw(download_params):
    """
    :return:
    """

    for item in utils.generate_input_parameters(download_params['D_states']):
        state, county, year, parameter, read_frequency = item
        r = None

        try:
            duration = map_read_frequency_to_duration[read_frequency]
        except Exception as e:
            print('Error: {}. Cant download this frequency. Using 1 hour duration'.format(e))
            duration = '1'

        curr_download_url = download_params[
                                'D_base_raw_url'] + "?user={M_USER}&pw={M_PASS}&format={M_FORMAT}&param={param}&bdate={year}0101&edate={year}1231&state={state}&county={county}&dur={duration}".format(
            M_USER=download_params['D_user'], M_PASS=download_params['D_pass'],
            M_FORMAT=download_params['D_format'], param=parameter, year=year, state=state, county=county,
            duration=duration)

        full_out_file_path = os.path.join(download_params['D_prefix'], state, county, str(year) + '_' + str(year),
                                          'frequency_' + read_frequency, parameter + '.csv')

        if utils.file_exists(full_out_file_path):
            print('File: {} already downloaded!'.format(full_out_file_path))
            continue
        try:
            # stream=True was being used so we could check the size already downloaded
            # and show a progress bar. Future implementation probably
            print('Downloading: {}'.format(full_out_file_path))

            r = requests.get(curr_download_url, stream=True)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.content.decode(r.encoding)))

                utils.write_dataframe_to_file(df, full_out_file_path)

                if r:
                    r.close()
            else:
                print("Error downloading file: ", item)
                exit()
        except Exception as e:
            print('Error downloading file: {}'.format(e))
            if r:
                r.close()
            continue
