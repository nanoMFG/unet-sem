##################################################################################################
# U-Net utils.py
####################################################################################################

# Append text to a log file
def append_to_log(text, directory='output', filename='out.log', carriage_return=True):
    os.makedirs(directory, exist_ok=True)
    if carriage_return:
        write = text + "\n"
    else:
        write = text

    with open(os.path.join(directory, filename), 'a') as f:
        f.write(write)

# Save U-Net model
def save_model_unet(model, epoch, accuracy, directory='output'):
    fname = 'model_E%003d_%s.hdf5' % (epoch, round(accuracy, 3))
    dirname = os.path.join(directory, '%003d_epoch' % epoch)
    os.makedirs(dirname, exist_ok=True)
    model.save(os.path.join(dirname, fname))
    return os.path.join(dirname, fname)

# Save output
def save_output(img, mask, output, epoch, index, directory='output'):
    os.makedirs(os.path.join(directory, '%003d_epoch' % epoch), exist_ok=True)

    with open(os.path.join(directory, '%003d_epoch' % epoch, '%003d_input_image.json' % index), 'w') as f:
        json.dump(img[..., 0].tolist(), f)
    with open(os.path.join(directory, '%003d_epoch' % epoch, '%003d_input_mask.json' % index), 'w') as f:
        json.dump(mask[..., 0].tolist(), f)
    with open(os.path.join(directory, '%003d_epoch' % epoch, '%003d_output_mask.json' % index), 'w') as f:
        json.dump(output[..., 0].tolist(), f)

def resize_preserve_aspect_ratio(image, target_size):
    """
    Resize an image preserving its aspect ratio.
    If either side is smaller than the target size, tile the image.
    
    :param image: Input PIL Image.
    :param target_size: Desired output size (width, height).
    :return: Resized and tiled image.
    """
    # Calculate the aspect ratio of the image.
    aspect = image.width / image.height
    
    # Initial resize dimensions.
    new_width = target_size[0]
    new_height = int(new_width / aspect)
    
    # Adjust width and height if exceeding target size.
    if new_height > target_size[1]:
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    # Resize while preserving the aspect ratio.
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Tile the image if it's smaller than the target size.
    tiled_img = Image.new("L", target_size)
    for x in range(0, target_size[0], new_width):
        for y in range(0, target_size[1], new_height):
            tiled_img.paste(image, (x, y))
    
    return tiled_img.crop((0, 0, target_size[0], target_size[1]))

def read_data(img_mask_path, target_size, preserve_aspect_ratio=False):
    img_path = img_mask_path[0]
    mask_path = img_mask_path[1]

    img = Image.open(img_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    img = np.array(img)
    img = img[..., np.newaxis]

    mask = np.array(mask)
    mask = mask[..., np.newaxis]

    return img, mask

# Random crop function
def random_crop(img, mask, random_crop_size=(256, 256), max_crop=False):
    height, width = img.shape[0], img.shape[1]
    if max_crop:
        dx = dy = min(height, width)
    else:
        dy, dx = random_crop_size
        dy = min(dy, height)
        dx = min(dx, width)
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :], mask[y:(y + dy), x:(x + dx), :]

# Apply contrast to an image
def contrast(img, scale):
    out_img = Image.fromarray(img[..., 0])
    contrast = ImageEnhance.Contrast(image)
    out_img = contrast.enhance(scale)

    return np.array(out_img)[..., np.newaxis]

# # Apply random augmentations
# def random_augment(img, mask, aug_dict):
#     image_datagen = ImageDataGenerator(**aug_dict)
#     seed = np.random.randint(1e9)

#     out_img = image_datagen.random_transform(img, seed=seed)
#     out_mask = image_datagen.random_transform(mask, seed=seed)

    return out_img, out_mask
def random_augment(img, mask, aug_dict):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    seed = np.random.randint(1e9)

    img_tensor = tf.expand_dims(img, 0)
    mask_tensor = tf.expand_dims(mask, 0)

    img_aug = image_datagen.flow(img_tensor, batch_size=1, seed=seed)
    mask_aug = image_datagen.flow(mask_tensor, batch_size=1, seed=seed)

    out_img = next(img_aug)[0]
    out_mask = next(mask_aug)[0]

    return out_img, out_mask

def resize_image(out_img,output_size):
    # Convert to PIL Image and resize
    out_img = Image.fromarray(out_img[..., 0])
    out_img = out_img.resize(output_size,Image.ANTIALIAS)

    # Convert back to numpy and normalize
    out_img = np.array(out_img)[..., np.newaxis]
    out_img = out_img / 255
    return out_img

def generate_batch(img, mask, batch_size=32, random_crop_size=(256, 256), output_size=(256, 256), 
                   crop=True, augment=True, aug_dict={}, max_crop=False):

    LX, LY, LZ = img.shape

    out_imgs = np.zeros((batch_size,) + output_size + (1,))
    out_masks = np.zeros((batch_size,) + output_size + (1,))
    for b in range(batch_size):
        seed = np.random.randint(1e9)

        out_img = img.copy()
        out_mask = mask.copy()

        # Apply cropping
        if crop:
            out_img, out_mask = random_crop(out_img, out_mask, random_crop_size, max_crop)

        # Apply data augmentation
        if augment:
            out_img, out_mask = random_augment(out_img, out_mask, aug_dict)
        
        out_img = resize_image(out_img,output_size)
        out_mask = resize_image(out_mask,output_size)

        out_imgs[b, ...] = out_img
        out_masks[b, ...] = out_mask

    return out_imgs, out_masks

def compute_num_imgs(train_paths, batch_size, augment, augmentation_factor, crop, crop_factor):
    num_imgs = len(train_paths)
    
    if augment:
        num_imgs += len(train_paths) * batch_size * augmentation_factor
#         num_imgs += len(train_paths) * augmentation_factor
        
    if crop:
        num_imgs += len(train_paths) * batch_size * crop_factor
#         num_imgs += len(train_paths) * crop_factor
        
    return num_imgs


def load_imgs_and_masks_with_augmentation(train_paths, input_size, batch_size, augment, augmentation_factor, data_gen_args, crop, crop_factor, crop_size, max_crop, preserve_aspect_ratio=False):
    """
    Load images and masks from the given training paths, and apply augmentation and preprocessing.

    Parameters:
    train_paths (list): A list of tuples containing the file paths for images and their corresponding masks.
    batch_size (int): The number of augmented images and masks to be generated from each input image.
    input_size (tuple): A tuple (height, width) representing the desired size of the output images and masks.
    crop_size (tuple): A tuple (height, width) representing the size of the random crop to be applied during augmentation.
    crop (bool): A flag indicating whether to apply cropping during augmentation.
    augment (bool): A flag indicating whether to apply data augmentation to the input images and masks.
    data_gen_args (dict): A dictionary containing the parameters for data augmentation.
    max_crop (int): The maximum number of attempts to find a valid crop.

    Returns:
    imgs (numpy.ndarray): A NumPy array containing the preprocessed and augmented images.
    masks (numpy.ndarray): A NumPy array containing the preprocessed and augmented masks corresponding to the images.

    Notes:
    - The dimensions of the input images and masks should be the same.
    - The input images and masks are assumed to be grayscale.
    """

    num_imgs = compute_num_imgs(train_paths, batch_size, augment, augmentation_factor, crop, crop_factor)
    
    imgs = np.zeros((num_imgs, input_size[0], input_size[1], 1))
    masks = np.zeros((num_imgs, input_size[0], input_size[1], 1))

    i_img = 0
    i_mask = 0
    for img_path, img_mask_path in enumerate(train_paths):
        img, mask = read_data(img_mask_path,(input_size[0], input_size[1]),preserve_aspect_ratio)
        imgs[i_img] = resize_image(img,input_size)
        masks[i_mask] = resize_image(mask,input_size)
        i_img += 1
        i_mask += 1
        if (augment):
            aug_imgs, aug_masks = generate_batch(
                img,
                mask,
                batch_size=augmentation_factor*batch_size,
                random_crop_size=crop_size,
                output_size=input_size,
                crop=False,
                augment=True,
                aug_dict=data_gen_args,
                max_crop=max_crop
            )
            # Can likely convert this to use slicing instead
            for aug_img in aug_imgs:
                imgs[i_img] = aug_img
                i_img += 1
            for aug_mask in aug_masks:
                masks[i_mask] = aug_mask
                i_mask += 1
        if (crop):
            aug_imgs, aug_masks = generate_batch(
                img,
                mask,
                batch_size=crop_factor*batch_size,
                random_crop_size=crop_size,
                output_size=input_size,
                crop=True,
                augment=False,
                aug_dict=data_gen_args,
                max_crop=max_crop
            )
            # Can likely convert this to use slicing instead
            for aug_img in aug_imgs:
                imgs[i_img] = aug_img
                i_img += 1
            for aug_mask in aug_masks:
                masks[i_mask] = aug_mask
                i_mask += 1
            
    return imgs, masks

def load_tb_images(test_paths, input_size,preserve_aspect_ratio=False):
    tb_images = np.zeros((len(test_paths), input_size[0], input_size[1], 1))
    i_tb = 0
    for img_path, img_mask_path in enumerate(test_paths):
        img, mask = read_data(img_mask_path,(input_size[0], input_size[1]))
        img = resize_image(img,input_size)
        mask = resize_image(mask,input_size)
        tb_images[i_tb] = img
        i_tb += 1
    return tb_images
   


def batch_generator(paths, input_size, augment, augmentation_factor, data_gen_args, crop, crop_factor, crop_size, max_crop, preserve_aspect_ratio, batch_size):
    # Number of original images needed to get a batch of size `batch_size` after augmentation
    effective_batch_size = batch_size // augmentation_factor
    
    while True:
        # Shuffle the paths
        shuffle(paths)
        # Loop over the paths in effective batches
        for i in range(0, len(paths), effective_batch_size):
            batch_paths = paths[i:i+effective_batch_size]
            # Load the images and masks for the current batch
            imgs, masks = load_imgs_and_masks_with_augmentation(
                batch_paths,
                input_size,
                batch_size,
                augment,
                augmentation_factor,
                data_gen_args,
                crop,
                crop_factor,
                crop_size,
                max_crop,
                preserve_aspect_ratio
            )
            # Only yield up to the intended batch_size
            for j in range(0, len(imgs), batch_size):
                yield imgs[j:j+batch_size], masks[j:j+batch_size]

def create_dataset_from_generator(paths, input_size, augment, augmentation_factor, data_gen_args, crop, crop_factor, crop_size, max_crop, preserve_aspect_ratio, batch_size):
    
    dataset = tf.data.Dataset.from_generator(
    batch_generator,
    args=(paths, input_size, augment, augmentation_factor, data_gen_args, crop, crop_factor, crop_size, max_crop, preserve_aspect_ratio, batch_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=((batch_size, input_size[0], input_size[1], 1), (batch_size, input_size[0], input_size[1], 1))
)
    
    # Shuffle and repeat the dataset for better training
    dataset = dataset.shuffle(buffer_size=100)  # you can adjust buffer size as needed
    dataset = dataset.repeat()  # repeat indefinitely
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # automatic prefetching for better performance
    
    return dataset