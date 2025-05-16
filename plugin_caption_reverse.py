import re
from typing import List
import ast
import numpy as np
from swift.plugin.orm import ORM, orms
from swift.utils import get_logger
from shapely.geometry import box
from shapely.ops import unary_union
from qwen_vl_utils import smart_resize

import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = get_logger()
    
class Format(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has the required format."""
        pattern = (
            r'^\s*<caption>.*?</caption>\s*'
            r'<bbox>\s*\[\[.*?\]\]\s*</bbox>\s*'
            r'<answer>.*?</answer>\s*$'
        )
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]

class CaptionAccuracy(ORM):
    def __call__(self, completions: List[str], image_caption: List[str], **kwargs) -> List[float]:
        """Reward function that evaluates the correctness of the generated caption."""
        pattern = r'<caption>(.*?)</caption>'
        scores = []
        
        # 初始化 BLEU smoothing
        smooth = SmoothingFunction().method1
        
        for completion, gt_caption in zip(completions, image_caption):
            match = re.search(pattern, completion, re.DOTALL)
            if not match:
                scores.append(0.0)  # 没有 <caption> 直接判 0
                continue

            pred_caption = match.group(1).strip()

            # 计算 BLEU-4 并加平滑
            reference = nltk.word_tokenize(gt_caption.lower())
            hypothesis = nltk.word_tokenize(pred_caption.lower())
            bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=smooth)  # 计算 BLEU-4
            
            # 计算 ROUGE-L
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(gt_caption, pred_caption)['rougeL'].fmeasure  # 取 ROUGE-L 的 F1 分数
            
            # 计算最终得分
            final_score = (bleu_score + rouge_score) / 2
            scores.append(final_score)

        return scores

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
        """
        计算所有预测框并集与所有真实框并集的 IoU。
        其中，每个框均为 [x1, y1, x2, y2]，并集可能不是规则的长方形，
        使用 shapely 进行精确计算。
        """
        # 将每个边界框转换为 shapely 的 box 对象
        pred_polygons = [box(b[0], b[1], b[2], b[3]) for b in pred_boxes]
        gt_polygons = [box(b[0], b[1], b[2], b[3]) for b in gt_boxes]

        # 计算各自的并集
        pred_union = unary_union(pred_polygons)
        gt_union = unary_union(gt_polygons)

        # 计算交集面积与并集面积
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
orms['external_caption_acc'] = CaptionAccuracy
orms['external_bbox_acc'] = BoundingBoxAccuracy
orms['extern_vqa_acc'] = VQAAccuracy