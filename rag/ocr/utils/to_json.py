# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Any, Dict, List, Tuple

import numpy as np

class ToJSON:
    @classmethod
    def to(
        cls, boxes: np.ndarray, txts: Tuple[str], scores: Tuple[float]
    ) -> List[Dict[Any, Any]]:
        results = []
        for box, txt, score in zip(boxes, txts, scores):
            results.append({"box": box.tolist(), "txt": txt, "score": score})
        return results

import json
def to_json(result) -> str:
    if hasattr(result, 'boxes') and hasattr(result, 'txts') and hasattr(result, 'scores'):
        data = ToJSON.to(result.boxes, result.txts, result.scores)
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps({}, ensure_ascii=False, indent=2)
