import mlflow
import socket
import time
import time
import socket
from reactxen.utils.model_inference import watsonx_llm

UNKNOWN = "unknown"


class GenAIInstructClient:
    def __init__(
        self,
        name,
        description,
        skill,
        model,
        params,
        system_message,
        question_message,
    ):
        self.name = name
        self.description = description
        self.skill = skill
        self.model = model
        self.system_message = system_message
        self.question_message = question_message
        self.params = params

        # track the request in and request out
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

        # trial
        self._max_retries = 3
        self._retry_delay = 10

    def _update_tokens_usage(
        self, prompt_tokens=0, completion_tokens=0, total_tokens=0
    ):
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._total_tokens += total_tokens

    def create(self, context, messages, experiment_id):
        """ """
        time.sleep(5)  # putting sleep for 5 second
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as conv:
            q_dict = {"Question": messages}
            mlflow.log_dict(q_dict, "Question.json")
            result = None
            for _ in range(1, self._max_retries + 1):
                try:
                    if self.question_message:
                        pprompt = ""
                        if len(messages) > 0:
                            pprompt = f"System Prompt: {self.system_message} \n\n {self.question_message} \n\n Question: {messages} Please use (Internal thought)."
                        else:
                            pprompt = f"System Prompt: {self.system_message}"
                        result = watsonx_llm(prompt=pprompt, **self.params)
                        print (result)
                    else:
                        pprompt = ""
                        if len(messages) > 0:
                            pprompt = f"System Prompt: {self.system_message} \n\n Question: {messages} Please use (Internal thought)."
                        else:
                            pprompt = f"System Prompt: {self.system_message}"
                        result = watsonx_llm(prompt=pprompt, **self.params)
                        print (result)
                    break
                except (OSError, socket.error, ConnectionResetError, Exception) as e:
                    print("Error ...." + str(e))
                    time.sleep(self._retry_delay)

            if result:
                a_dict = {"Answer": result["generated_text"]}
                self._update_tokens_usage(
                    result["input_token_count"],
                    result["input_token_count"] + result["generated_token_count"],
                    result["generated_token_count"],
                )
                mlflow.log_dict(a_dict, "Answer.json")
                return result["generated_text"]
            else:
                return ""

    def print_token_usage(self):
        print(
            f"The usages Promt Token: {self._prompt_tokens}, \
              Generated Token: {self._completion_tokens}, Total Token : {self._total_tokens}"
        )
