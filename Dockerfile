FROM nvcr.io/nvidia/tritonserver:24.09-py3

CMD tritonserver --model-repository=/triton/models --strict-model-config=false