system_prompt_template = """#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{{"task": task, "id": task_id, "dep": dependency_task_id, "args": {{"text": text or <GENERATED>-dep_id, "image": image_url or <GENERATED>-dep_id, "audio": audio_url or <GENERATED>-dep_id}}}}]. The special tag "<GENERATED>-dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The "args" field must in ["timeseries", "text", "image", "audio"], nothing else. The task MUST be selected from the following options: {tools}. There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need to reply empty JSON []."""
system_template =        """#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{{"task": task, "id": task_id, "dep": dependency_task_id, "args": {{"input name": text may contain <resource-dep_id>}}}}]. The special tag "dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The task MUST be selected from the following tools (along with tool description, input name and output type): {tools}. There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need to reply empty JSON []."""  # noqa: E501

modified_system_prompt_template = """#1 Task Planning Stage: The AI assistant can parse 
user input to several tasks: [{{"task": task, "id": task_id, "dep": dependency_task_id, 
"args": {{'data": text or <resource-dep_id>, "text": text or <resource-dep_id>, "image": image_url or <resource-dep_id>, 
"audio": audio_url or <resource-dep_id>}}}}]. The special tag "<resource-dep_id>" refer 
to the one generated text/image/audio in the dependency task (Please consider whether the 
dependency task generates resources of this type.) and "dep_id" must be in "dep" list. 
The "dep" field denotes the ids of the previous prerequisite tasks which generate a new 
resource that the current task relies on. The "args" field must in ["timeseries", "text", 
"image", "audio"], nothing else. 

The task MUST be selected from the following options: {tools}. There may be multiple tasks of the same type. Think step by step about all the 
tasks needed to resolve the user's request. Parse out as few tasks as possible while 
ensuring that the user request can be resolved. Pay attention to the dependencies and 
order among tasks. If the user input can't be parsed, you need to reply empty JSON [].

"""

final_system_prompt = """#1 Task Planning Stage: The AI assistant parses user input into a set of tasks, each represented by a JSON object. Each task has an ID and may depend on the results of other tasks. Tasks are executed sequentially based on their dependencies.

## Task Structure:
Each task is represented as a JSON object with the following fields:
```json
{
  "task": "task_name",
  "id": task_id,
  "dep": [dependency_task_ids],  // List of task IDs that must complete first.
  "args": {                      // Arguments for the task.
    "data": "text or <resource-dep_id>",  // The data used for the task. "<resource-dep_id>" refers to a resource produced by a dependency task.
    "text": "text or <resource-dep_id>",  // Text data for the task.
    "image": "image_url or <resource-dep_id>", // Image URL or a resource from a previous task.
    "audio": "audio_url or <resource-dep_id>"  // Audio URL or a resource from a previous task.
  }
}

The task MUST be selected from the following options: {tools}. There may be multiple tasks of the same type. Think step by step about all the 
tasks needed to resolve the user's request. Parse out as few tasks as possible while 
ensuring that the user request can be resolved. Pay attention to the dependencies and 
order among tasks. If the user input can't be parsed, you need to reply empty JSON [].
"""
