# Локальная проверка в условиях платформы: Python 3.12 и версии из requirements.txt организаторов.
# docker build -t alfa-agent . && docker run --rm --network none -e AGENT_TIME_BUDGET=600 \
#   -v "$PWD/../data/Регрессия:/task" alfa-agent /task/task.md
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.12.0 pandas==2.2.3 scikit-learn==1.5.2 \
    lightgbm==4.6.0 catboost==1.2.8 xgboost==2.1.4 pydantic==2.10.6
WORKDIR /app
COPY solution.py agent_*.py ./
ENTRYPOINT ["python", "solution.py"]
