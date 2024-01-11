from optimum.exporters.tasks import TasksManager

model_type = "gpt-neo"

# For instance, for the ONNX export.

backend = "onnx"

distilbert_tasks = list(TasksManager.get_supported_tasks_for_model_type(model_type, backend).keys())

print(distilbert_tasks)