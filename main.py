from model import *
from utils import *


def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = K.cast(sh[0] / n_gpus,'int32')

    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=2):
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])

    towers = []

    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)
#         merged = merge(towers, mode='concat', concat_axis=0)

    return Model(input=[x], output=merged)

    ###############################################################
################ INPUT BLOCK ##################################
###############################################################

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
n_gpus = 2
num_epochs = 5
batch_size =32
image_mask_paths = [("data/image<%d>.tif"%i,"data/image_mask<%d>.jpg"%i) for i in range(1,41)]
split = 0.1
random_crop_size = (256,256)

###############################################################
###############################################################

num_images = len(image_mask_paths)
test_paths = image_mask_paths[:int(split*num_images)]
train_paths = image_mask_paths[int(split*num_images):]

model = unet(input_size=random_crop_size+(1,))
if n_gpus > 1:
    model = to_multi_gpu(model,n_gpus=n_gpus)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
for epoch in range(num_epochs):
    shuffle(train_paths)
    for i, img_mask_path in enumerate(train_paths):
        img, mask = read_data(img_mask_path)
        aug_imgs, aug_masks = augment_input(img,mask,data_gen_args,random_crop_size=random_crop_size)
        loss = model.train_on_batch(aug_imgs,aug_masks)
        print("epoch: %d (%d/%d), %s: %s"%(epoch,i,len(train_paths),model.metrics_names,loss))

    test_loss = []
    for i, img_mask_path in enumerate(test_paths):
        img, mask = read_data(img_mask_path)
        out_imgs = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
        out_masks = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
        for b in range(batch_size):
            seed = np.random.randint(1e9)
            crop_img, crop_mask = random_crop(img,mask,random_crop_size)
        out_imgs[b,...] = image_datagen.random_transform(crop_img,seed=seed)
        out_masks[b,...] = image_datagen.random_transform(crop_mask,seed=seed)

        test_loss.append(model.test_on_batch(out_imgs,out_masks))
    test_loss = np.mean(np.array(test_loss),axis=0)
    print("epoch: %d, %s: %s"%(epoch,model.metrics_names,test_loss))
