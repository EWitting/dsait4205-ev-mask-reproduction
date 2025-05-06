from src.dvs_config import *
from src.dvs_dataset import *

config = DvsConfig()
config.display()

# Training dataset
dataset_train = RGBDDataset()
dataset_train.load('../data/N_MNIST_images_20ms_skip_50', 'training')
dataset_train.prepare()

# Validation dataset
dataset_testing = RGBDDataset()
dataset_testing.load('../data/N_MNIST_images_20ms_skip_50', 'testing')
dataset_testing.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 3)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, last, none

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask", "conv1"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
elif init_with == "none":
    print('Not loading any weights!')

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

####################### UNCOMMENT THESE WHEN TRAINING #######################

# # Train the head branches
# # Passing layers="heads" freezes all layers except the head
# # layers. You can also pass a regular expression to select
# # which layers to train by name pattern.
model.train(dataset_train, dataset_testing,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')
#
# # model.train(dataset_train, dataset_testing,
# #             learning_rate=config.LEARNING_RATE / 10,
# #             epochs=12,
# #             layers="all")
#
# # # Save weights
# # # Typically not needed because callbacks save after every epoch
# # # Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_dvs_2ep.h5")
model.keras_model.save_weights(model_path)

model.train(dataset_train, dataset_testing,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='heads')
#
# # # Save weights
# # # Typically not needed because callbacks save after every epoch
# # # Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_dvs_5ep.h5")
model.keras_model.save_weights(model_path)

print('\n\n---------------------------------------------------------------')
print('---------------------------------------------------------------')
print('-------------------- Vroom vroom tuning!!! --------------------')
print('---------------------------------------------------------------')
print('---------------------------------------------------------------\n\n')


# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.

model.train(dataset_train, dataset_testing,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=15,
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_dvs.h5")
model.keras_model.save_weights(model_path)

print(model)

####################### UNCOMMENT ABOVE WHEN TRAINING #######################
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


class InferenceConfig(DvsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()



# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_testing.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_testing, inference_config,
                           image_id) # , use_mini_mask=False

log("original_image", original_image)
print(original_image.shape)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)
print(results)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_train.class_names, scores=r['scores'])