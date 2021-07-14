import copy
import csv
import json
import os
import utils
import pandas as pd

# ------------------- EXTREMELLY USEFUL FUNCTIONS THAT RETRIEVE ALL THE DATA + METADATAUNIT
def generate_cleaned_data_paths(country, state, county, station, start_year, end_year, poc='', filetype='csv'):
    """
    Very important method that retrieves all the files to be read to recover the data and metadata
    :param country:
    :param state:
    :param county:
    :param station:
    :param start_year:
    :param end_year:
    :param parameter:
    :param poc: parameter occurrence code (happens inside a station)
    :param filetype: json (metadata) or csv (data)
    :yield: relative path to file (sorted)
    """
    root_dir = os.path.join("datasets", "cleaned_datasets", state, county)

    for curr_year in range(int(start_year), int(end_year) + 1):
        curr_year = str(curr_year)
        temp_root_dir = os.path.join(root_dir, curr_year + '_' + curr_year, 'frequency_1H')
        filter_by = country + state + county + station + poc
        for aval_file in os.listdir(temp_root_dir):
            if poc != '':
                # retrieves data for an specific station + poc (specific file)
                if aval_file.split('_')[0] == filter_by and aval_file.split('.')[-1] == filetype:
                    # print(aval_file)
                    yield os.path.join(temp_root_dir, aval_file)
            else:
                # retrieves all data for one station (no poc)
                if aval_file[0:12] == filter_by and aval_file.split('.')[-1] == filetype:
                    # print(aval_file)
                    yield os.path.join(temp_root_dir, aval_file)

    return

def retrieve_specific_data_df_csv(country, state, county, station, start_year='1997', end_year='2018'):
    """
    :param country:
    :param state:
    :param county:
    :param station:
    :param parameter:
    :param poc:
    :param start_year:
    :param end_year:
    :return:
    """
    filetype = 'csv'
    df = pd.DataFrame()

    # get data for parameter
    for path in generate_cleaned_data_paths(country=country, state=state, county=county, station=station,
                                            poc='', start_year=start_year, end_year=end_year, filetype=filetype):
        try:
            temp_df = pd.read_csv(path, index_col=0)
            poc = os.path.split(path)[1].split('_')[0][12:]
            temp_df.rename(columns=lambda x: x + '-' + poc, inplace=True)
            df = df.append(temp_df, sort=True)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(e)
            continue
    return df

def extract_years_ranges(in_dict, string_dict, metainfo):
    """
    :param in_dict:
    :param string_dict:
    :param parameter:
    :param metainfo:
    :return:
    """
    for year_info in metainfo[string_dict]:
        try:
            if year_info[1] not in in_dict.keys():
                in_dict[year_info[1]] = []
            in_dict[year_info[1]].append(year_info[0])
        except Exception as e:
            print(e)
            continue

    return in_dict

def retrieve_specific_data_metadata_json(country, state, county, station, start_year='1997', end_year='2018'):
    filetype = 'json'
    all_metainfo_template = {
        'lat': {},
        'lon': {},
        'unit': {},
        'method_code': {},
        'GISDatum': {},
        'data_status': {},
        'elev': {},
        'mpc': {},
        'mpc_value': {},
        'uncertainty': {},
        'qc': {},
        'qualifiers': {}
    }
    all_metainfo = {}

    # get data for parameter
    for path in generate_cleaned_data_paths(country=country, state=state, county=county, station=station,
                                            poc='', start_year=start_year, end_year=end_year, filetype=filetype):
        # retrieves all metadata for given arguments
        try:
            poc = os.path.split(path)[1].split('_')[0][12:]
            with open(path, 'r') as f:
                metainfo = json.loads(f.read())
                for param in metainfo.keys():
                    # print(param)
                    n_uid = param + '-' + poc
                    if n_uid not in all_metainfo.keys():
                        all_metainfo[n_uid] = copy.deepcopy(all_metainfo_template)
                    # print(all_metainfo)
                    for key in all_metainfo_template.keys():
                        # print(key)
                        # print(metainfo)
                        all_metainfo[n_uid][key] = extract_years_ranges(all_metainfo[n_uid][key], key, metainfo[param])
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(e)
            continue
    return all_metainfo

def get_all_stations(county, state='06'):
    available_stations = {}
    try:
        # filename = state + county + '_stations.json'
        filename = 'stations_list.json'
        full_in_path = os.path.join("datasets", "cleaned_datasets", state, county, filename)
        with open(full_in_path, 'r') as f:
            available_stations = json.loads(f.read())
    except FileNotFoundError as e:
        print('File not found.. wrong args')
    return available_stations

def get_all_counties(state='06'):
    # we can just list the directory too...
    available_counties_dict = {
        "019": "Fresno",
        "029": "Kern",
        "037": "Los Angeles",
    }

    # return render_template('update_available_counties_station_view.html.jinja2', available_counties_dict=available_counties_dict)
    for county in available_counties_dict.keys():
        yield county

def retrieve_unit_content(unit):
    # translates unit code to unit name
    file_path = os.path.join('datasets', 'AQS_code_list', 'code_to_unit.csv')
    input_file = csv.reader(open(file_path))
    for row in input_file:
        n_key, n_value = row
        if float(n_key) == float(unit):
            return 'Unit name: {}'.format(n_value), n_value

    return '', ''

# def metadata_retrieve_only_difference(all_metainfo):
#     """
#     Here we remove the yearly values and only store when the values change
#     :param all_metainfo:
#     :return:
#     """
#     cleaned_metainfo = {}
#     for key, values in all_metainfo.items():
#         suppres_repeated_values = ''
#         for value in values:
#             actual_value = str(value[1])
#             if suppres_repeated_values != actual_value:
#                 cleaned_metainfo[key][actual_value].append()
#                 suppres_repeated_values = actual_value
#     return all_metainfo

@utils.timeit
def get_station_data_usepa(given_parameters, country, state):
    # this method will be based on the one from automl research.
    for county in get_all_counties():
        # print(county)
        for station in get_all_stations(county):
            # print(station)
            df = retrieve_specific_data_df_csv(country=country, state=state, county=county, station=station)
            all_metainfo = retrieve_specific_data_metadata_json(country=country, state=state, county=county, station=station)
            if not utils.is_dataframe_ok(df):
                continue

            df = df.sum(level=0, min_count=1) # so we merge the indexes form different poc files
            yield df, all_metainfo, country, state, county, station

@utils.timeit
def get_all_missing_percentage_info_usepa(given_parameters):
    # get all the columns for each year
    # percentage of missing data per column
    root_dir = given_parameters['R_input_dataset_path']
    info_dict = {}

    # you must lsit fir and take itens ending in _meta.json
    for full_in_file_path in utils.get_info_refine(given_parameters, root_dir):
        year_year, uid, full_path = full_in_file_path
        uid = uid.split('_')[0]
        try:
            # have to open the file and start extracting the info
            with open(full_path, 'r') as f:
                # print(f.read())
                content = json.loads(f.read())
                for key, value in content.items():
                    if uid in info_dict.keys():
                        # means that already initialized that station (it already has at least one info record in the dict)
                        if key in info_dict[uid].keys():
                            pass
                        else:
                            info_dict[uid][key] = {}
                    else:
                        # initialize list to add the years and stats
                        info_dict[uid] = {}
                        info_dict[uid][key] = {}
                    #     MUST use key somewher.. figure out where
                    curr_year_stats = {
                        value['missing_percentage'][0]: {
                            'missing_percentage': value['missing_percentage'][1],
                        }
                    }

                    info_dict[uid][key].update(curr_year_stats)
        except Exception as e:
            print('Error: {}'.format(e))
            continue

    return info_dict

@utils.timeit
def get_dataframe_kuwait(given_parameters):
    # for data.. we yield
    input_path = given_parameters['R_input_dataset_path']
    for file in os.listdir(input_path):
        df = pd.read_csv(os.path.join(input_path, file), header=0, index_col=0)
        if not utils.is_dataframe_ok(df):
            continue
        yield df, file
    return
