import numpy as np
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from scipy.ndimage import label
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from typing import List, Tuple, Union

class ConnectedComponentCropPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.verbose = verbose


    """
    基于最大连通域的裁剪预处理器
    """
    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        # 1. 调用默认预处理加载原始数据
        data, seg, properties = super().run_case(image_files, seg_file, plans_manager,configuration_manager,dataset_json)

        # 2. 找到最大连通域的 bbox
        if seg is not None:
            mask = seg > 0
        else:
            mask = data.sum(axis=0) > 0  # 如果没有标签，用图像本身

        labeled, num_features = label(mask)
        # print("num_features", num_features)
        if num_features == 0:
            bbox = [slice(0, s) for s in mask.shape]
        else:
            largest_cc = labeled == (np.bincount(labeled.flat)[1:].argmax() + 1)
            bbox = self.get_bbox_from_mask(largest_cc)


        # 3. 裁剪图像和标签
        try:
            data_cropped = data[(slice(None),) + tuple(bbox)]
            seg_cropped = seg[(slice(None),) + tuple(bbox)] if seg is not None else None
        except:
            print("[ERROR]")
            return data, seg, properties
        # print(f"data shape:{data.shape}------->data_cropped:{data_cropped.shape}")
        # print(f"seg shape:{seg.shape}------->seg_cropped:{seg_cropped.shape}")
        # print(f"bbox:{[slice(0, s) for s in data.shape[1:]]}------->{bbox}")
        # print("data shape:", data.shape)
        # print("seg shape:", seg.shape if seg is not None else None)
        # print("bbox:", bbox)

        # 4. 检查裁剪后标签是否为空
        if seg is not None and seg_cropped.sum() == 0:
            print(f"[WARNING] Empty segmentation after crop , skipping crop.")
            data_cropped = data
            seg_cropped = seg
            bbox = [slice(0, s) for s in data.shape[1:]]
        else:
            print("[INFO] cropping segmentation ...")

        # 5. 更新结果
        data = data_cropped
        seg = seg_cropped
        properties['crop_bbox'] = bbox
        properties['original_shape_after_crop'] = data.shape[1:]

        return data, seg, properties

    @staticmethod
    def get_bbox_from_mask(mask: np.ndarray) -> List[slice]:
        """根据二值掩膜计算 bounding box（仅空间维度）"""
        coords = np.array(np.nonzero(mask))
        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1) + 1
        # 只返回 3 个 slice，对应 Z, Y, X
        bbox = [slice(min_i, max_i) for min_i, max_i in zip(min_coords[-3:], max_coords[-3:])]
        return bbox