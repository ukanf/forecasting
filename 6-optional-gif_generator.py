import imageio
import os
absolute_shared_folder = os.path.join('//winnas5', 'd$', 'ProjectData', 'ML_Forecast', 'KUWAIT')
refined_test_full_path = os.path.join(absolute_shared_folder, 'models_runs', 'kuwait')
parameters= ['realpredict-O3_t+']

input_path = os.path.join(refined_test_full_path)
output_path = os.path.join(input_path)
# common_prefix_file = parameter_to_run
t_range_to_gif = range(1, 24)
kargs = { 'duration': 0.5 }


for curr_folder in os.listdir(input_path):
    for parameter_to_run in parameters:
        images = []
        for t in t_range_to_gif:
            filename = os.path.join(input_path, curr_folder, parameter_to_run + str(t) + '.png')
            try:
                images.append(imageio.imread(filename))
            except FileNotFoundError:
                print('Error opening image: {}. Will not save gif'.format(filename))
                break
        if len(images) > 0:
            imageio.mimsave(os.path.join(output_path, curr_folder, parameter_to_run + '-realpredict.gif'), images, 'GIF', **kargs)
            print('Saved: {}'.format(os.path.join(output_path, curr_folder, parameter_to_run + '-realpredict.gif')))
