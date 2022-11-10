# unet-sem
image segmentation for CVD graphene SEM images  
Based on: https://github.com/zhixuhao/unet

# Usage

    main.py [-h] [--crop] [--shuffle_data] [--augment_after AUGMENT_AFTER]
               [--augment] [--max_crop] [--crop_size CROP_SIZE]
               [--input_size INPUT_SIZE] [--ngpu NGPU] [--nepochs NEPOCHS]
               [--batch_size BATCH_SIZE] [--split SPLIT] [--lr LR]
               [--input_dir INPUT_DIR]

    optional arguments:
      -h, --help            show this help message and exit
      --crop                Constructs batch using random crops.
      --shuffle_data        Shuffles data paths to switch what is in test/train.
      --augment_after AUGMENT_AFTER
                            Start augmenting data ater specified epoch,
                            inclusively.
      --augment             Constructs batch using augmentations.
      --max_crop            Crops using the maximum square size for each image.
                            Crop size is ignored.
      --crop_size CROP_SIZE
                            Size of cropped sample.
      --input_size INPUT_SIZE
                            Model input size. Cropped images will be rescaled to
                            this size.
      --ngpu NGPU           Number of GPUs.
      --nepochs NEPOCHS     Number of epochs.
      --batch_size BATCH_SIZE
                            Number of samples per batch.
      --split SPLIT         If float, fraction of data to use for validation. If
                            integer, number of folds. If zero, train on all data
                            (used for final model.
      --lr LR               Learning rate.
      --input_dir INPUT_DIR
                            Directory to pull images from

# Image Data

Images used to train the neural network are available in the `data` directory:
- All directories that end in `_Bad` contain low fidelity masks and their corresponding images
- All directories that end in `_Good` contain high fidelity masks and their corresponding images
- The `Old_data` directory also contains high fidelity masks and their corresponding images

Use the `fileproc.py` script to combine all the images into a single directory and rename them correctly
