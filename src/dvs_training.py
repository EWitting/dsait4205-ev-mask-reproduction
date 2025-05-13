from src.dvs_config import *
from src.dvs_dataset import *
from src.dvs_dataset_multiple import *
import hashlib

config = DvsConfig()
inference_config = InferenceConfig()
config.display()

def obj_to_hash(obj):
    return hashlib.md5(str(obj).encode()).hexdigest()

def show_train_samples(dataset, n):
    image_ids = np.random.choice(dataset.image_ids, n)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

def load_weights(model, pretrained_weights):
    if pretrained_weights == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif pretrained_weights == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", "conv1"])
    elif pretrained_weights == "last":
        model.load_weights(model.find_last(), by_name=True)
    else:
        print('Not loading any weights!')

def train_model(data_path, fit_params, multiple_digits=False, pretrained_weights='coco', skip_if_exists=False):
    """Train the model with the given fit parameters. Model is saved under name based on the data path and fit parameters, and whether multiple digits are used.
    
    :data_path: Path to the data
    :fit_params: dict or list[dict], with parameters to pass to keras model.fit, e.g. {'epochs': 10, 'learning_rate': 0.001}. If list, will run training for each dict in the list.
    :multiple_digits: Wether to use the dataset that automatically generates samples with multiple digits
    :pretrained_weights: Pretrained weights
    :skip_if_exists: If True, will skip training if the model already exists
    
    Returns:
    :model_path: Path to the model
    """
    model_string = f'mask_rcnn_dvs_{obj_to_hash(data_path)}_{obj_to_hash(fit_params)}'
    if multiple_digits:
        model_string += '_multi'
    model_path = os.path.join(MODEL_DIR, f'{model_string}.h5')
    if skip_if_exists and os.path.exists(model_path):
        print(f'Model {model_path} already exists, skipping training')
        return model_path

    dataset_class = RGBDDatasetMultiple if multiple_digits else RGBDDataset

    # Training dataset
    dataset_train = dataset_class()
    dataset_train.load(data_path, 'training')
    dataset_train.prepare()

    # Validation dataset
    dataset_testing = dataset_class()
    dataset_testing.load(data_path, 'testing')
    dataset_testing.prepare()

    # Load and display random samples
    show_train_samples(dataset_train, n=3)

    # Initialize model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    load_weights(model, pretrained_weights)

    # Train model for one or multiple stages
    if isinstance(fit_params, dict):
        fit_params = [fit_params]
    for i, fit_param in enumerate(fit_params):
        print(f'Training with {fit_param}, stage {i+1} of {len(fit_params)}')
        model.train(dataset_train, dataset_testing, **fit_param)

    # Save the model
    model.keras_model.save_weights(model_path)
    print(f'Model saved to {model_path}')
    
    # Run inference to visualize some results
    inference_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    inference_model.load_weights(model_path, by_name=True)

    # Test and visualize a random image
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

    results = inference_model.detect([original_image], verbose=1)
    print(results)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_train.class_names, scores=r['scores'])
    
    return model_path
