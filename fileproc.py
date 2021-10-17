import os
import argparse
import shutil

parser = argparse.ArgumentParser()

# parser.add_argument('--num_sets', default=4, type=int, help='How many sets there are available, one-indexed')
# parser.add_argument('--use_sets', default='all', type = str, help='Either \'all\' or a string of which numbers, i.e. \'124\'')
# parser.add_argument('--use_good_bad', default='both', type=str, help='One of: good, bad, both')
parser.add_argument('--input_dir_list', default='data', type=str, help='One string of all directory names separated by commas')
parser.add_argument('--output_dir', default='combined_data', type=str)

args = vars(parser.parse_args())

# if args['use_sets'] == 'all':
#     sets = [i for i in range(1, args['num_sets'] + 1)]
# else:
#     sets = []
#     for num in args['use_sets']:
#         if(int(num) <= args['num_sets']):
#             sets.append(int(num))

input_dir_list = args['input_dir_list'].split(',')
for i in range(len(input_dir_list)):
    input_dir_list[i] = input_dir_list[i].strip()

if os.path.isdir(args['output_dir']):
    shutil.rmtree(args['output_dir'])
    os.mkdir(args['output_dir'])
else:
    os.mkdir(args['output_dir'])

# for set in sets:
#     if args['use_good_bad'] == 'good' or args['use_good_bad'] == 'both':
#         for file in os.listdir(os.path.join(args['input_dir'], 'Set_{}_Good'.format(set))):
#             shutil.copy(os.path.join(os.path.join(args['input_dir'], 'Set_{}_Good'.format(set)), file), args['output_dir'])
#     if args['use_good_bad'] == 'bad' or args['use_good_bad'] == 'both':
#         for file in os.listdir(os.path.join(args['input_dir'], 'Set_{}_Bad'.format(set))):
#             shutil.copy(os.path.join(os.path.join(args['input_dir'], 'Set_{}_Bad'.format(set)), file), args['output_dir'])

count = 1
fail_list = []

for directory in input_dir_list:
    i = 1
    while i != 0:
        imagep = os.path.join(directory,"image<"+str(i)+">.tif")
        maskp = os.path.join(directory,"image_mask<"+str(i)+">.png")
        imagep_new = os.path.join(args['output_dir'],"image_"+str(count)+".tif")
        maskp_new = os.path.join(args['output_dir'],"image_mask_"+str(count)+".png")

        try:
            shutil.copy(imagep, imagep_new)
            try:
                shutil.copy(maskp, maskp_new)
                i += 1
                count += 1
            except:
                fail_list = fail_list.append(imagep)
                i += 1
        except:
            i = 0

print("\n {} images and their corresponding masks copied".format(count-1))
print("\n List of images with no masks: ", fail_list)
