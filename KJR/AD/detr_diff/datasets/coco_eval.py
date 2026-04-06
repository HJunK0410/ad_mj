# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        
        # Store predictions for v1-style precision/recall calculation
        self.all_predictions = []  # List of (boxes, scores, labels, image_id)
        self.all_targets = []  # List of (boxes, labels, image_id)

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        # Store predictions for v1-style calculation
        for img_id, prediction in predictions.items():
            if len(prediction) > 0:
                boxes = prediction["boxes"].cpu().numpy()
                scores = prediction["scores"].cpu().numpy()
                labels = prediction["labels"].cpu().numpy()
                self.all_predictions.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                    'image_id': img_id
                })
                
                # Get ground truth for this image
                ann_ids = self.coco_gt.getAnnIds(imgIds=img_id)
                anns = self.coco_gt.loadAnns(ann_ids)
                if len(anns) > 0:
                    gt_boxes = []
                    gt_labels = []
                    for ann in anns:
                        # Convert from COCO format (x, y, w, h) to (x1, y1, x2, y2)
                        x, y, w, h = ann['bbox']
                        gt_boxes.append([x, y, x + w, y + h])
                        gt_labels.append(ann['category_id'] - 1)  # Convert to 0-based
                    if len(gt_boxes) > 0:
                        self.all_targets.append({
                            'boxes': np.array(gt_boxes, dtype=np.float32),
                            'labels': np.array(gt_labels, dtype=np.int64),
                            'image_id': img_id
                        })

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
    
    def compute_precision(self, iou_type='bbox', iou_threshold=0.5):
        """
        Compute precision at a specific IoU threshold.
        Precision = TP / (TP + FP) at IoU threshold.
        
        Args:
            iou_type: 'bbox' or 'segm'
            iou_threshold: IoU threshold (default: 0.5)
        
        Returns:
            precision value (float)
        """
        if iou_type not in self.coco_eval:
            return 0.0
        
        coco_eval = self.coco_eval[iou_type]
        
        # Find the index of the IoU threshold
        if not hasattr(coco_eval, 'params') or not hasattr(coco_eval.params, 'iouThrs'):
            return 0.0
        
        iou_thrs = coco_eval.params.iouThrs
        if iou_threshold not in iou_thrs:
            # Find closest threshold
            iou_idx = np.argmin(np.abs(np.array(iou_thrs) - iou_threshold))
        else:
            iou_idx = list(iou_thrs).index(iou_threshold)
        
        # Get evaluation results
        if not hasattr(coco_eval, 'evalImgs') or coco_eval.evalImgs is None:
            return 0.0
        
        eval_imgs = coco_eval.evalImgs
        if len(eval_imgs) == 0:
            return 0.0
        
        # Calculate TP and FP at the specified IoU threshold
        # evalImgs is a list of dicts, each containing evaluation results
        # We need to extract TP/FP information at the specified IoU threshold
        tp_sum = 0
        fp_sum = 0
        
        for eval_img in eval_imgs:
            if eval_img is None:
                continue
            
            # Get detections at the specified IoU threshold
            if 'dtMatches' in eval_img and len(eval_img['dtMatches']) > iou_idx:
                dt_matches = eval_img['dtMatches'][iou_idx]
                # dtMatches contains matched ground truth indices (TP) or -1 (FP)
                tp = np.sum(dt_matches >= 0)
                fp = np.sum(dt_matches == -1)
                tp_sum += tp
                fp_sum += fp
        
        # Calculate precision
        if tp_sum + fp_sum == 0:
            return 0.0
        
        precision = tp_sum / (tp_sum + fp_sum)
        return float(precision)
    
    def compute_recall(self, iou_type='bbox', iou_threshold=0.5):
        """
        Compute recall at a specific IoU threshold.
        Recall = TP / (TP + FN) at IoU threshold.
        
        Args:
            iou_type: 'bbox' or 'segm'
            iou_threshold: IoU threshold (default: 0.5)
        
        Returns:
            recall value (float)
        """
        if iou_type not in self.coco_eval:
            return 0.0
        
        coco_eval = self.coco_eval[iou_type]
        
        # Find the index of the IoU threshold
        if not hasattr(coco_eval, 'params') or not hasattr(coco_eval.params, 'iouThrs'):
            return 0.0
        
        iou_thrs = coco_eval.params.iouThrs
        if iou_threshold not in iou_thrs:
            # Find closest threshold
            iou_idx = np.argmin(np.abs(np.array(iou_thrs) - iou_threshold))
        else:
            iou_idx = list(iou_thrs).index(iou_threshold)
        
        # Get evaluation results
        if not hasattr(coco_eval, 'evalImgs') or coco_eval.evalImgs is None:
            return 0.0
        
        eval_imgs = coco_eval.evalImgs
        if len(eval_imgs) == 0:
            return 0.0
        
        # Calculate TP and FN at the specified IoU threshold
        tp_sum = 0
        fn_sum = 0
        
        for eval_img in eval_imgs:
            if eval_img is None:
                continue
            
            # Get detections and ground truth at the specified IoU threshold
            if 'dtMatches' in eval_img and len(eval_img['dtMatches']) > iou_idx:
                dt_matches = eval_img['dtMatches'][iou_idx]
                # dtMatches contains matched ground truth indices (TP) or -1 (FP)
                tp = np.sum(dt_matches >= 0)
                tp_sum += tp
            
            if 'gtMatches' in eval_img and len(eval_img['gtMatches']) > iou_idx:
                gt_matches = eval_img['gtMatches'][iou_idx]
                # gtMatches contains matched detection indices (TP) or -1 (FN)
                fn = np.sum(gt_matches == -1)
                fn_sum += fn
        
        # Calculate recall
        if tp_sum + fn_sum == 0:
            return 0.0
        
        recall = tp_sum / (tp_sum + fn_sum)
        return float(recall)
    
    def compute_precision_recall_v1(self, iou_type='bbox', eps=1e-16):
        """
        Compute precision and recall using v1 (YOLO) style calculation.
        This method:
        1. Sorts predictions by confidence
        2. Computes TP/FP at multiple IoU thresholds (0.5:0.05:0.95)
        3. Creates Precision-Recall curves
        4. Selects precision/recall at the point where F1 score is maximum
        
        Args:
            iou_type: 'bbox' or 'segm'
            eps: Small value to avoid division by zero
        
        Returns:
            tuple: (precision, recall) at max F1 point
        """
        if len(self.all_predictions) == 0 or len(self.all_targets) == 0:
            return 0.0, 0.0
        
        # Collect all predictions and targets
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []
        
        # Create image_id to target mapping
        img_to_targets = {t['image_id']: t for t in self.all_targets}
        
        for pred in self.all_predictions:
            img_id = pred['image_id']
            if img_id in img_to_targets:
                target = img_to_targets[img_id]
                all_pred_boxes.append(pred['boxes'])
                all_pred_scores.append(pred['scores'])
                all_pred_labels.append(pred['labels'])
                all_gt_boxes.append(target['boxes'])
                all_gt_labels.append(target['labels'])
        
        if len(all_pred_boxes) == 0:
            return 0.0, 0.0
        
        # Concatenate all predictions and targets
        pred_boxes = np.concatenate(all_pred_boxes, axis=0)
        pred_scores = np.concatenate(all_pred_scores, axis=0)
        pred_labels = np.concatenate(all_pred_labels, axis=0)
        gt_boxes = np.concatenate(all_gt_boxes, axis=0)
        gt_labels = np.concatenate(all_gt_labels, axis=0)
        
        # Build mapping from prediction index to image_id
        pred_to_img = []
        for pred in self.all_predictions:
            img_id = pred['image_id']
            if img_id in img_to_targets:
                n_preds = len(pred['boxes'])
                pred_to_img.extend([img_id] * n_preds)
        
        # Build mapping from ground truth index to image_id
        gt_to_img = []
        for target in self.all_targets:
            img_id = target['image_id']
            n_gts = len(target['boxes'])
            gt_to_img.extend([img_id] * n_gts)
        
        # Sort by confidence (descending)
        sorted_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_idx]
        pred_scores = pred_scores[sorted_idx]
        pred_labels = pred_labels[sorted_idx]
        pred_to_img = np.array(pred_to_img)[sorted_idx]
        
        # IoU thresholds: 0.5:0.05:0.95 (10 thresholds)
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        # Compute TP for each IoU threshold
        tp = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)
        
        # Track matched ground truths per image
        matched_gt_per_img = {}
        for img_id in set(gt_to_img):
            img_gt_mask = np.array(gt_to_img) == img_id
            matched_gt_per_img[img_id] = np.zeros(np.sum(img_gt_mask), dtype=bool)
        
        # For each prediction, check if it matches any ground truth
        for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            pred_img_id = pred_to_img[i]
            
            # Get ground truths for this image
            img_gt_mask = np.array(gt_to_img) == pred_img_id
            img_gt_boxes = gt_boxes[img_gt_mask]
            img_gt_labels = gt_labels[img_gt_mask]
            img_matched = matched_gt_per_img[pred_img_id]
            
            if len(img_gt_boxes) == 0:
                continue
            
            # Find matching ground truths with same class and not yet matched
            matching_mask = (img_gt_labels == pred_label) & (~img_matched)
            if not np.any(matching_mask):
                continue
            
            matching_gt_boxes = img_gt_boxes[matching_mask]
            
            # Compute IoU with all matching ground truths
            ious = self._compute_iou_single(pred_box, matching_gt_boxes)
            
            # For each IoU threshold, check if there's a match
            for j, iou_thresh in enumerate(iou_thresholds):
                if np.any(ious >= iou_thresh):
                    tp[i, j] = True
                    # Mark the best matching ground truth as matched
                    best_match_idx = np.argmax(ious)
                    original_gt_idx = np.where(matching_mask)[0][best_match_idx]
                    img_matched[original_gt_idx] = True
                    break
        
        # Use IoU threshold 0.5 (first threshold) for precision/recall calculation
        tp_iou50 = tp[:, 0]
        
        # Count total ground truths
        n_gt = len(gt_boxes)
        
        if n_gt == 0:
            return 0.0, 0.0
        
        # Compute cumulative TP and FP
        tpc = tp_iou50.cumsum()
        fpc = (~tp_iou50).cumsum()
        
        # Compute precision and recall curves
        precision_curve = tpc / (tpc + fpc + eps)
        recall_curve = tpc / (n_gt + eps)
        
        # Compute F1 curve
        f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + eps)
        
        # Find max F1 index
        max_f1_idx = np.argmax(f1_curve)
        
        precision = float(precision_curve[max_f1_idx])
        recall = float(recall_curve[max_f1_idx])
        
        return precision, recall
    
    def _compute_iou_single(self, box1, boxes2, eps=1e-7):
        """
        Compute IoU between a single box and multiple boxes.
        
        Args:
            box1: Single box [x1, y1, x2, y2]
            boxes2: Multiple boxes [N, 4] with [x1, y1, x2, y2]
            eps: Small value to avoid division by zero
        
        Returns:
            IoU values [N]
        """
        # Compute intersection
        x1_min = np.maximum(box1[0], boxes2[:, 0])
        y1_min = np.maximum(box1[1], boxes2[:, 1])
        x2_max = np.minimum(box1[2], boxes2[:, 2])
        y2_max = np.minimum(box1[3], boxes2[:, 3])
        
        inter_area = np.maximum(0, x2_max - x1_min) * np.maximum(0, y2_max - y1_min)
        
        # Compute union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = box1_area + boxes2_area - inter_area
        
        # Compute IoU
        iou = inter_area / (union_area + eps)
        return iou

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # DETR 모델의 labels는 0부터 시작하지만, COCO는 category_id가 1부터 시작
            # 따라서 +1을 해줘야 ground truth와 매칭됨
            # (YOLO 데이터셋도 COCO API 변환 시 cls_id + 1로 변환했으므로 일관성 유지)
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k] + 1,  # 0-based -> 1-based (COCO format)
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
