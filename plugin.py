import re
from typing import List
import ast
import numpy as np
from swift.plugin.orm import ORM, orms
from swift.utils import get_logger
from shapely.geometry import box
from shapely.ops import unary_union
from qwen_vl_utils import smart_resize

logger = get_logger()
    
class Format(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<bbox>.*?</bbox>\s*<answer>.*?</answer>(?![\s\S])'
        #print(completions)
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]

class BoundingBoxAccuracy(ORM):
    def __call__(self, completions: List[str], bboxs: List[List[List[float]]], width: List[int], height: List[int],  **kwargs) -> List[float]:
        """Reward function that checks if the completion has the correct bboxs."""
        pattern = r'<bbox>(.*?)</bbox>'
        scores = []
        #print(bboxs)
        for completion, gt_bboxes, w, h in zip(completions, bboxs, width, height):
            match = re.search(pattern, completion, re.DOTALL)
            if not match:
                scores.append(0.0)
                continue
            try:
                pred_bboxes = ast.literal_eval(match.group(1))
                if isinstance(pred_bboxes, list) and all(not isinstance(x, list) for x in pred_bboxes):
                    pred_bboxes = [pred_bboxes]
                if not isinstance(pred_bboxes, list) or any(
                    not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(num, (int, float)) for num in bbox))
                    for bbox in pred_bboxes
                ):
                    scores.append(0.0)
                    continue
            except (SyntaxError, ValueError):
                scores.append(0.0)
                continue
            
            input_height,input_width = smart_resize(h, w, min_pixels=200704, max_pixels=401408)

            scale_x = w / input_width
            scale_y = h / input_height

            pred_bboxes = self.resize_bboxes(pred_bboxes, scale_x, scale_y)

            # Convert lists to numpy arrays for easier comparison
            pred_bboxes = np.array(pred_bboxes)
            gt_bboxes = np.array(gt_bboxes)
            
            # if pred_bboxes.shape != gt_bboxes.shape:
            #     scores.append(0.0)
            #     continue
            
            # Compute IoU for each bounding box pair and average
            score = self.union_iou(pred_bboxes, gt_bboxes)
            scores.append(np.mean(score))
            
        return scores
    
    def resize_bboxes(self, bboxes: List[List[float]], scale_x: float, scale_y: float) -> List[List[float]]:
        return [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in bboxes]

    def union_iou(self, pred_boxes: List[List[float]], gt_boxes: List[List[float]]) -> float:
        pred_polygons = [box(b[0], b[1], b[2], b[3]) for b in pred_boxes]
        gt_polygons = [box(b[0], b[1], b[2], b[3]) for b in gt_boxes]

        pred_union = unary_union(pred_polygons)
        gt_union = unary_union(gt_polygons)

        intersection_area = pred_union.intersection(gt_union).area
        union_area = pred_union.union(gt_union).area

        return intersection_area / (union_area + 1e-6)

class VQAAccuracy(ORM):
    def __call__(self, completions: List[str], possible_answers: List[List[str]], **kwargs) -> List[float]:
        """Reward function that checks if the completion contains a possible answer."""
        pattern = r'<answer>(.*?)</answer>'
        scores = []
        
        for completion, _answer in zip(completions, possible_answers):
            match = re.search(pattern, completion, re.DOTALL)
            if not match:
                scores.append(0.0)
                continue
            
            answer_extract = match.group(1).strip()
            scores.append(1.0 if answer_extract in _answer else 0.0)
        
        return scores

orms['external_format'] = Format
orms['external_bbox_acc'] = BoundingBoxAccuracy
orms['extern_vqa_acc'] = VQAAccuracy