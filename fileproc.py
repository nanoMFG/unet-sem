import os
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--num_sets', default=4, type=int, help='How many sets there are available, one-indexed')
parser.add_argument('--use_sets', default='all', type = str, help='Either \'all\' or a string of which numbers, i.e. \'124\'')
parser.add_argument('--use_good_bad', default='both', type=str, help='One of: good, bad, both')
parser.add_argument('--input_dir', default='data', type=str)
parser.add_argument('--output_dir', default='combined_data', type=str)

args = vars(parser.parse_args())

if args['use_sets'] == 'all':
    sets = [i for i in range(1, args['num_sets'] + 1)]
else:
    sets = []
    for num in args['use_sets']:
        if(int(num) <= args['num_sets']):
            sets.append(int(num))

if args['output_dir'] not in os.listdir():
    os.mkdir(args['output_dir'])
else:
    shutil.rmtree(args['output_dir'])
    os.mkdir(args['output_dir'])

for set in sets:
    if args['use_good_bad'] == 'good' or args['use_good_bad'] == 'both':
        for file in os.listdir(os.path.join(args['input_dir'], 'Set_{}_Good'.format(set))):
            shutil.copy(os.path.join(os.path.join(args['input_dir'], 'Set_{}_Good'.format(set)), file), args['output_dir'])
    if args['use_good_bad'] == 'bad' or args['use_good_bad'] == 'both':
        for file in os.listdir(os.path.join(args['input_dir'], 'Set_{}_Bad'.format(set))):
            shutil.copy(os.path.join(os.path.join(args['input_dir'], 'Set_{}_Bad'.format(set)), file), args['output_dir'])
