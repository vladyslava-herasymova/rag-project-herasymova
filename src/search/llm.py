import json
import re
import sys
import time
from typing import Optional, Type, Union
from pydantic import BaseModel, ValidationError
import ollama

MODEL = "llama3.2"
NUM_CTX = 8192
MAX_TOKENS = 2048
TIMEOUT = 300
STATS = {"calls": 0, "seconds": 0.0}

def _call_ollama(prompt: str, schema: Optional[dict], temperature: float) -> str:
    client = ollama.Client(timeout=TIMEOUT)
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"options": {"temperature": temperature, "num_ctx": NUM_CTX, "num_predict": MAX_TOKENS}}
    if schema is not None:
        kwargs["format"] = schema                  
    try:
        resp = client.chat(model=MODEL, messages=messages, **kwargs)
        text, done_reason = resp.message.content, getattr(resp, "done_reason", None)
            
    except Exception as e:  # noqa: BLE001
        if "timeout" in type(e).__name__.lower():
            raise TimeoutError(f"no answer from the model within {TIMEOUT:.0f} s (raise LLM_TIMEOUT to wait longer)") from e
        raise
    if done_reason == "length":
        raise SystemError(f"the model hit the {MAX_TOKENS}-token limit before finishing its answer "
                       f"(it may be looping; see raw_model_output)", text)
    return text


def _call(prompt: str, schema: Optional[dict], temperature: float) -> str:
    t0 = time.time()
    try:
        return _call_ollama(prompt, schema, temperature)
    finally:
        STATS["calls"] += 1
        STATS["seconds"] += time.time() - t0

def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()

def get_response(prompt: str, schema: Union[Type[BaseModel], dict], temperature: float = 0.0):
    is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
    json_schema = schema.model_json_schema() if is_model else schema
    raw = _call(prompt, json_schema, temperature)
    text = _strip_fences(raw)
    try:
        return (schema.model_validate_json(text) if is_model else json.loads(text)), raw
    except (ValidationError, json.JSONDecodeError) as e:
        raise ValueError(f"model output does not match the schema: {e}", raw) from e