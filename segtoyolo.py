from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
convert_segment_masks_to_yolo_seg("/home/lenovo/data/liujiaji/Datasets-Multiview/COD10K-v3/Test/GT_Object", 
                                  "/home/lenovo/data/liujiaji/Datasets-Multiview/COD10K-v3/txt",
                                    classes=69)
