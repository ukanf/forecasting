import csv
import itertools
import json
import os
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from handle_data.utils import utils
import numpy as np
import data_retriever as dr

# ------------------- MISSING PERCENTAGE HEATMAP
def vis_heatmap_missing_percentage(df, aux_group, vis_parameters):
    """
    by Felipe Ukan - (c) 2019 Lakes Env. Software
    Creates heat map to visualize the data availability distribution
    :param df:
    :param aux_group:
    :param vis_parameters:
    :return:
    """
    sns.set(style="darkgrid")

    f, ax = plt.subplots(figsize=(9, 9))

    # if aux_group == 1:
    country_code = df.columns.values[0].split('_')[1][0:3]
    state_code = df.columns.values[0].split('_')[1][3:5]
    county_code = df.columns.values[0].split('_')[1][5:8]
    poc = df.columns.values[0].split('_')[1][12:]
    map_code_to_name = utils.load_code_to_name_dicts(vis_parameters)

    map_code_to_site = map_code_to_name[state_code][county_code]['sites']
    map_code_to_parameter = map_code_to_name[state_code][county_code]['parameters']
    state_name = map_code_to_name[state_code]['name']
    county_name = map_code_to_name[state_code][county_code]['name']

    if aux_group == 1:
        title_switcher = map_code_to_site
        x_axis_switcher = map_code_to_parameter
    else:
        title_switcher = map_code_to_parameter
        x_axis_switcher = map_code_to_site

    uid = df.columns.values[0].split('_')[aux_group]
    print(uid)
    try:
        if aux_group == 1:
            title_str = '{} - {}. {} - {} \n {} - {}'.format(state_code, state_name, county_code, county_name,
                                                             uid[8:12], title_switcher[uid[8:12]])
            df.rename(columns=lambda x: x.split('_')[((aux_group + 1) % 2)] + '-' + x.split('_')[1][12:], inplace=True)
        else:
            title_str = '{} - {}. {} - {} \n {} - {}'.format(state_code, state_name, county_code, county_name, uid,
                                                             title_switcher[uid])
            df.rename(
                columns=lambda x: x.split('_')[((aux_group + 1) % 2)][8:12] + '-' + x.split('_')[((aux_group + 1) % 2)][
                                                                                    12:], inplace=True)
        ax.set_title(title_str)

    except KeyError:
        ax.set_title(uid)
        df.rename(columns=lambda x: x.split('_')[((aux_group + 1) % 2)], inplace=True)

    # df = df.reindex(index=df.index[::-1])

    g = sns.heatmap(df, vmin=0, vmax=100, cbar_kws={'label': '% of missing data'})
    # g.cax.set_position([.15, .2, .03, .45])

    # print(country_code + county_code + uid)
    # print(df.columns)
    f.tight_layout()
    # plt.show()
    if aux_group == 1:
        plt.savefig(os.path.join('missing_data_explorer', 'stations', 'missing_perncent_' + uid[:12]))
    else:
        plt.savefig(os.path.join('missing_data_explorer', 'parameters', 'missing_perncent_' + country_code + state_code + county_code + uid))

    plt.close()

def vis_missing_percentage(vis_parameters):
    """
    by Felipe Ukan - (c) 2019 Lakes Env. Software
    Prepares data to send to vis_heatmap_missing_percentage in order to create heat
    matrix with missing percentage per station or parameter
    :param info_refine:
    :param vis_parameters:
    :return:
    """
    info_refine = dr.get_all_missing_percentage_info_usepa(vis_parameters)

    stats_df = pd.DataFrame(index=reversed(vis_parameters['R_year_range']))
    for uid, missing_data in info_refine.items():
        for param, year_stats in missing_data.items():
            param_station = param + '_' + uid
            for year, stats in year_stats.items():
                stats_df.loc[year, param_station] = stats['missing_percentage']

    stats_df.fillna(100, inplace=True)

    # always grouping by parameter or station
    if vis_parameters['R_o_visualize_missing_percentage_group_by'] == 'station':
        aux_group = 1
        aux_groups = set(map(lambda x: x.split('_')[aux_group], stats_df.columns))
        # following loop works well for station
        for group_column in aux_groups:
            grouped_df = pd.DataFrame()
            for column in stats_df.columns:
                if column.split('_')[aux_group][:12] == group_column[:12]:
                    grouped_df[column] = stats_df[column]
                    print(column)
            if len(grouped_df.columns) > 0:
                vis_heatmap_missing_percentage(grouped_df, aux_group, vis_parameters)
    elif vis_parameters['R_o_visualize_missing_percentage_group_by'] == 'parameter':
        aux_group = 0
        aux_groups = set(map(lambda x: x.split('_')[aux_group], stats_df.columns))
        # following loop works well for station
        all_counties_prefix = set(map(lambda x: x.split('_')[(aux_group + 1) % 2][:8], stats_df.columns))
        for county_prefix in all_counties_prefix:
            for group_column in aux_groups:
                grouped_df = pd.DataFrame()
                for column in stats_df.columns:
                    if column.split('_')[(aux_group + 1) % 2][:8] == county_prefix:
                        if column.split('_')[aux_group] == group_column:
                            grouped_df[column] = stats_df[column]
                if len(grouped_df.columns) > 0:
                    vis_heatmap_missing_percentage(grouped_df, aux_group, vis_parameters)
    else:
        # by default it will be grouped by parameter
        print('Wrong R_o_visualize_missing_percentage_group_by. Using parameter')
        aux_group = 0

# ------------------- JOINTPLOT
def  vis_jointplot_marginal(vis_parameters):
    state = '06'
    country = '840'
    for df, all_metainfo, country, state, county, station in dr.get_station_data_usepa(vis_parameters, country, state):
        print(country, state, county, station)
        try:
            metainfo_unit = {}
            # check if unit is uniq?
            for key, value in all_metainfo.items():
                for in_key, in_values in all_metainfo[key].items():
                    if in_key != 'unit':
                        print('we just extract the unit for now... data is still in all_metainfo.. so change here to retrieve the rest of the values')
                        continue
                    if len(set(in_values.keys())) == 1:
                        # here we know unit is unique and we just extract the unit.. the rest is still on all_metainfo and we might want to use later
                        content, unique_unit = dr.retrieve_unit_content(list(set(in_values.keys()))[0])
                        metainfo_unit[key] = unique_unit
            sns.set(style="ticks")
            for parameter1, parameter2 in itertools.combinations(df.columns, 2):
                df1 = df[parameter1].dropna()
                df2 = df[parameter2].dropna()
                df_plot = pd.concat([df1, df2], axis=1, sort=False)
                df_plot = df_plot.dropna()
                jointplot = sns.jointplot(df_plot[parameter1], df_plot[parameter2], kind="hex", color="#4CB391")
                code_to_param_path = os.path.join('datasets', 'AQS_code_list', 'code_to_parameter_min.json')
                with open(code_to_param_path, 'r') as f:
                    code_to_param = json.loads(f.read())
                    only_parameter1, poc1 = parameter1.split('-')
                    only_parameter2, poc2 = parameter2.split('-')
                    x_label = str(code_to_param[only_parameter1]) + ' - ' + str(metainfo_unit[parameter1]) + '  - poc:' + poc1
                    y_label = str(code_to_param[only_parameter2]) + ' - ' + str(metainfo_unit[parameter2]) + '  - poc:' + poc2
                jointplot.set_axis_labels(x_label, y_label, fontsize=12)
                fileid = 'jointplot' + country + state + county + station + '_' + parameter1 + '_' + parameter2 + '.png'
                curr_path = os.path.join('jointplot', state, county, station)
                if not os.path.exists(curr_path):
                    os.makedirs(curr_path)
                plt.savefig(os.path.join('jointplot', state, county, station, fileid))
                # plt.show()
                plt.close()
        except ValueError:
            print('Data range available for both parameters is do not meet. Both parameters must have the same amount of data for the same years available to generate the hex plot')
            continue
        except ZeroDivisionError:
            print('Not enought data to generate hex plot. Given parameters have no datapoint in common.')
            continue

# ------------------- CORRELATION MATRIX
def vis_correlation_matrix(vis_parameters):
    state = '06'
    country = '840'
    for df, all_metainfo, country_code, state_code, county_code, station_code in dr.get_station_data_usepa(vis_parameters, country, state):
        print(country_code, state_code, county_code, station_code)
        if len(df.columns) <= 1:
            print('Not enough columns for corr matrix')
            continue
        try:
            map_code_to_name = utils.load_code_to_name_dicts(vis_parameters)
            map_code_to_site = map_code_to_name[state_code][county_code]['sites']
            map_code_to_parameter = map_code_to_name[state_code][county_code]['parameters']
            state_name = map_code_to_name[state_code]['name']
            county_name = map_code_to_name[state_code][county_code]['name']
            title_str = '{} - {}. {} - {} \n {} - {}'.format(state_code, state_name, county_code, county_name, station_code, map_code_to_site[station_code])

            sns.set(style="darkgrid")
            corr = df.corr()
            for item in corr.columns:
                corr[map_code_to_parameter[item.split('-')[0]] + '-' + item.split('-')[1]] = corr[item]
                del corr[item]
            # Generate a mask for the upper triangle
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            f, ax = plt.subplots(figsize=(9, 9))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            sns.heatmap(corr, cmap=cmap, ax=ax, mask=mask, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
                        cbar_kws={'label': 'Correlation'})
            ax.set_title(title_str)
            f.tight_layout()

            fileid = 'corr_' + country_code + state_code + county_code + station_code + '.png'
            curr_path = os.path.join('correlation_matrix', state_code, county_code)
            if not os.path.exists(curr_path):
                os.makedirs(curr_path)
            plt.savefig(os.path.join('correlation_matrix', state_code, county_code, fileid))
            plt.close()
        except ValueError:
            print(
            'Data range available for both parameters is do not meet.')
            continue


