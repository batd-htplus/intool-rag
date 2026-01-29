import numpy as np

class ToMarkdown:
    @classmethod
    def to(cls, boxes, txts) -> str:
        """
        Restore text layout to Markdown based on OCR result box coordinates.

        Args:
            boxes (np.ndarray): Detected text boxes.
            txts (tuple): Recognized text strings.

        Returns:
            str: Markdown string simulating the original layout.
        """
        if boxes is None or txts is None:
            return "No text detected."

        # Bind box and text, then sort by top y and left x
        combined_data = sorted(
            zip(boxes, txts),
            key=lambda item: (
                cls.get_box_properties(item[0])["top"],
                cls.get_box_properties(item[0])["left"],
            ),
        )

        output_lines = []
        if not combined_data:
            return ""

        current_line_parts = [combined_data[0][1]]
        prev_props = cls.get_box_properties(combined_data[0][0])

        for box, text in combined_data[1:]:
            current_props = cls.get_box_properties(box)

            # Heuristic: decide layout
            min_height = min(current_props["height"], prev_props["height"])
            centers_are_close = abs(
                current_props["center_y"] - prev_props["center_y"]
            ) < (min_height * 0.5)

            overlap_top = max(prev_props["top"], current_props["top"])
            overlap_bottom = min(prev_props["bottom"], current_props["bottom"])
            has_vertical_overlap = overlap_bottom > overlap_top

            is_same_line = centers_are_close or has_vertical_overlap

            if is_same_line:
                current_line_parts.append("   ")  # Use spaces for separation
                current_line_parts.append(text)
            else:
                output_lines.append("".join(current_line_parts))
                vertical_gap = current_props["top"] - prev_props["bottom"]
                if vertical_gap > prev_props["height"] * 0.7:
                    output_lines.append("")  # New paragraph
                current_line_parts = [text]
            prev_props = current_props

        output_lines.append("".join(current_line_parts))
        return "\n".join(output_lines)

    @staticmethod
    def get_box_properties(box: np.ndarray) -> dict:
        """
        Calculate geometric properties from box coordinates.
        box shape is (4, 2) -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        ys = box[:, 1]
        xs = box[:, 0]

        top = np.min(ys)
        bottom = np.max(ys)
        left = np.min(xs)

        return {
            "top": top,
            "bottom": bottom,
            "left": left,
            "height": bottom - top,
            "center_y": top + (bottom - top) / 2,
        }

def to_markdown(result):
    """
    Convert RapidOCR result to markdown string.
    result: object with 'boxes' and 'txts' attributes
    """
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    return ToMarkdown.to(boxes, txts)
