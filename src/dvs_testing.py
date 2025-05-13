import os
import numpy as np
from src.dvs_config import *
from src.dvs_dataset import *
from src.dvs_dataset_multiple import *

inference_config = InferenceConfig()
config = DvsConfig()
config.display()

def test_model(model_path, dataset_path, multiple_digits=False, visualize_num=0):

    # Testing dataset
    dataset_validation = RGBDDatasetMultiple() if multiple_digits else RGBDDataset()
    dataset_validation.load(dataset_path, 'validation')

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                            config=inference_config,
                            model_dir=MODEL_DIR)
    model.load_weights(model_path, by_name=True)
    print("MODEL")
    print('\n\n\n')

    # Visualize some images with predictions
    for i in range(visualize_num):
        # Test on a random image
        image_id = random.choice(dataset_validation.image_ids)
        print(image_id)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_validation, inference_config,
                                image_id) # , use_mini_mask=False

        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    dataset_validation.class_names, figsize=(8, 8))

        results = model.detect([original_image], verbose=1)

        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                    dataset_validation.class_names, scores=r['scores'])

        empty_image = np.zeros(original_image.shape)

        visualize.display_instances(empty_image, r['rois'], r['masks'], r['class_ids'],
                                    dataset_validation.class_names, scores=r['scores'])


    # # Compute VOC-Style mAP @ IoU=0.5
    image_ids = dataset_validation.image_ids
    APs, ACCs, IoUs = [], [], []
    for image_id in image_ids:
    
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_validation, inference_config,
                                image_id)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        ap, precisions, recalls, overlaps, ious = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0)
    
        accuracy = utils.compute_accuracy(r['masks'], gt_mask)
        if len(ious) > 0:
            IoUs.append(np.mean(ious))
        else:
            IoUs.append(0.0)
        APs.append(ap)
        ACCs.append(accuracy)

    results = {
        'AP': np.mean(APs),
        'IoU': np.mean(IoUs),
        'Accuracy': np.mean(ACCs)
    }
    for metric, value in results.items():
        print(f"{metric}: {value}")

    return results