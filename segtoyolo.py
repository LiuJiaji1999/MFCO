from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
convert_segment_masks_to_yolo_seg("/home/lenovo/data/liujiaji/Datasets-Multiview/NC4K/GT", 
                                  "/home/lenovo/data/liujiaji/Datasets-Multiview/NC4K/txt",
                                    classes=80)
