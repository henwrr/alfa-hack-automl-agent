"""Фейковый OpenAI-совместимый сервер для проверки LLM-цикла без ключа.

Запуск: python tests/fake_llm_server.py 8765
Отвечает по очереди: битый текст -> валидный JSON с экспериментами -> stop.
Последний запрос сохраняет в tests/last_llm_request.json — удобно смотреть, что видит LLM.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ANSWERS = [
    "Сейчас подумаю... без JSON",
    json.dumps({
        "analysis": "Фейковый анализ: основная ошибка в хвосте таргета и в отдельных сегментах.",
        "experiments": [
            {"hypothesis": "Индикаторы пропусков + log1p таргета", "expected": "-1 RMSE",
             "spec": {"extends_best": True, "features": [{"op": "missing_flags"}]}},
            {"hypothesis": "Битая операция — хост должен вернуть понятную ошибку", "expected": "ошибка",
             "spec": {"extends_best": True, "features": [{"op": "ratio", "pairs": [["no_such_col", "x"]]}]}},
            {"hypothesis": "custom-признак через transform(df)", "expected": "проверка custom",
             "spec": {"extends_best": True, "features": [{"op": "custom", "code": "import numpy as np\ndef transform(df):\n    num = df.select_dtypes('number')\n    df['row_nan_count'] = num.isna().sum(axis=1)\n    return df"}]}},
        ],
        "stop": False,
    }, ensure_ascii=False),
    json.dumps({"analysis": "Идеи исчерпаны", "experiments": [], "stop": True}, ensure_ascii=False),
]
calls = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Path(__file__).with_name("last_llm_request.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
        answer = ANSWERS[min(calls["n"], len(ANSWERS) - 1)]
        calls["n"] += 1
        payload = {"choices": [{"message": {"role": "assistant", "content": answer}}],
                   "usage": {"total_tokens": sum(len(m["content"]) for m in body["messages"]) // 3}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1] if len(sys.argv) > 1 else 8765)), Handler).serve_forever()
