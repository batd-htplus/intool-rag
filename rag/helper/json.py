import json
import re
from typing import Union

def sanitize_json(response: Union[str, dict, list]) -> Union[dict, list, None]:
    if isinstance(response, (dict, list)):
        return response

    if not response or not isinstance(response, str):
        return None

    text = response.strip()

    if "```" in text:
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    text = re.sub(r"^[^{}\[\]]*\|\s*", "", text, flags=re.MULTILINE)

    obj_start = min(
        [i for i in [text.find("{"), text.find("[")] if i != -1],
        default=-1
    )
    obj_end = max(text.rfind("}"), text.rfind("]"))

    if obj_start == -1 or obj_end == -1 or obj_end <= obj_start:
        raise ValueError("No JSON object found in response")

    text = text[obj_start:obj_end + 1]

    return json.loads(text)
