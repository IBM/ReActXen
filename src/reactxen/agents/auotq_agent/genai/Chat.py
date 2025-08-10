from typing import Callable, Dict, List, Optional, Tuple, Union
import re
import json
import mlflow
import socket
import time
import os
from reactxen.utils.model_inference import watsonx_llm_chat

UNKNOWN = "unknown"


def content_str(content: Union[str, List]) -> str:
    if type(content) is str:
        return content
    rst = ""
    for item in content:
        if item["type"] == "text":
            rst += item["text"]
        else:
            assert (
                isinstance(item, dict) and item["type"] == "image_url"
            ), "Wrong content format."
            rst += "<image>"
    return rst


CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"


def extract_code(
    text: Union[str, List],
    pattern: str = CODE_BLOCK_PATTERN,
    detect_single_line_code: bool = False,
) -> List[Tuple[str, str]]:
    """Extract code from a text.

    Args:
        text (str or List): The content to extract code from. The content can be
            a string or a list, as returned by standard GPT or multimodal GPT.
        pattern (str, optional): The regular expression pattern for finding the
            code block. Defaults to CODE_BLOCK_PATTERN.
        detect_single_line_code (bool, optional): Enable the new feature for
            extracting single line code. Defaults to False.

    Returns:
        list: A list of tuples, each containing the language and the code.
          If there is no code block in the input text, the language would be "unknown".
          If there is code block but the language is not specified, the language would be "".
    """
    text = content_str(text)
    if not detect_single_line_code:
        match = re.findall(pattern, text, flags=re.DOTALL)
        return match if match else [(UNKNOWN, text)]

    # Extract both multi-line and single-line code block, separated by the | operator
    # `([^`]+)`: Matches inline code.
    code_pattern = re.compile(CODE_BLOCK_PATTERN + r"|`([^`]+)`")
    code_blocks = code_pattern.findall(text)

    # Extract the individual code blocks and languages from the matched groups
    extracted = []
    for lang, group1, group2 in code_blocks:
        if group1:
            extracted.append((lang.strip(), group1.strip()))
        elif group2:
            extracted.append(("", group2.strip()))

    # print (extracted)
    return extracted


class GenAIChatClient:
    def __init__(
        self,
        name,
        description,
        skill,
        model,
        params,
        system_message,
        post_process_text=False,
    ):
        self.name = name
        self.description = description
        self.skill = skill
        self._conversation_id = None
        self.system_message = system_message
        self.post_process_text = post_process_text
        self.model = model
        self.params = params

        # track the request in and request out
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

        # trial
        self._max_retries = 3
        self._retry_delay = 10

    def update_system_message(self, asset_class, asset_desc):
        self.system_message = f"{self.system_message} \n\n Asset Class: {asset_class} \n\n Asset Description: {asset_desc} \n\n"
        print(self.system_message)

    def _update_tokens_usage(
        self, prompt_tokens=0, completion_tokens=0, total_tokens=0
    ):
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._total_tokens += total_tokens

    def create(self, context, messages, experiment_id):
        """ """
        # todo: fix llm call
        time.sleep(5)  # putting sleep for 5 second
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as conv:
            q_dict = {"Question": messages}
            mlflow.log_dict(q_dict, "Question.json")
            if self._conversation_id:
                result = None
                for _ in range(1, self._max_retries + 1):
                    print([self.system_message] + messages)
                    try:
                        result = watsonx_llm_chat(
                            prompt=[self.system_message] + messages,
                            is_system_prompt=True,
                            **self.params,
                        )
                        break
                    except (
                        OSError,
                        socket.error,
                        ConnectionResetError,
                        Exception,
                    ) as e:
                        print("Error ...." + str(e))
                        time.sleep(self._retry_delay)

                if result:
                    a_dict = {"Answer": result['message']['content']}
                    # t_dict = result.generations[0][0].generation_info["token_usage"]
                    # self._update_tokens_usage(
                    #     t_dict["prompt_tokens"],
                    #     t_dict["completion_tokens"],
                    #     t_dict["total_tokens"],
                    # )
                    mlflow.log_dict(a_dict, "Answer.json")
                    if self.post_process_text:
                        return self.clean_user_assistant(result['message']['content'])
                    return result['message']['content']
                else:
                    return ""
            else:
                for _ in range(1, self._max_retries + 1):
                    result = None
                    print([self.system_message] + messages)
                    try:
                        result = watsonx_llm_chat(
                            prompt=[self.system_message] + messages,
                            is_system_prompt=True,
                            **self.params,
                        )
                        break
                    except (
                        OSError,
                        socket.error,
                        ConnectionResetError,
                        Exception,
                    ) as e:
                        print("Error ...." + str(e))
                        time.sleep(self._retry_delay)

                if result:
                    a_dict = {"Answer": result['message']['content']}
                    mlflow.log_dict(a_dict, "Answer.json")
                    if self.post_process_text:
                        return self.clean_user_assistant(result['message']['content'])
                    return result['message']['content']
                else:
                    return ""

    def clean_user_assistant(self, text):
        lines = text.split("\n")
        cleaned_text = "\n".join(
            line
            for line in lines
            if not (line.startswith("User:") or (line == "Assistant:"))
        )
        return cleaned_text

    def extract_questions(self, text):
        """Doing implmenetation

        :param text: _description_
        :type text: _type_
        :return: _description_
        :rtype: _type_
        """
        chat_agent_response = content_str(text)
        questions_start_index = chat_agent_response.find("1. ")

        if "1. " not in chat_agent_response:
            if "* " in chat_agent_response:
                questions_start_index = chat_agent_response.find("* ")
            elif "- " in chat_agent_response:
                questions_start_index = chat_agent_response.find("- ")
            elif "+ " in chat_agent_response:
                questions_start_index = chat_agent_response.find("+ ")

        if "\n\n" in chat_agent_response:
            questions_end_index = chat_agent_response.rfind("\n\n") + 2
        else:
            questions_end_index = len(chat_agent_response)

        if questions_end_index == questions_start_index:
            if "\n" in chat_agent_response:
                questions_end_index = chat_agent_response.rfind("\n")
            else:
                questions_end_index = len(chat_agent_response)

        questions_string = chat_agent_response[
            questions_start_index:questions_end_index
        ]

        # Splitting the questions into a list
        questions_list = questions_string.split("\n")

        # Removing empty elements from the list
        questions_list = [
            question.strip() for question in questions_list if question.strip()
        ]
        final_questions = []
        for item in questions_list:
            if len(item) > 5:
                first_space_index = item.find(" ")
                if first_space_index != -1:
                    final_questions.append(item[first_space_index + 1 :])
                else:
                    final_questions.append(item)

        # few more options - add latter

        # Printing the list of questions
        return final_questions

    def print_token_usage(self):
        print(
            f"The usages Promt Token: {self._prompt_tokens}, \
              Generated Token: {self._completion_tokens}, Total Token : {self._total_tokens}"
        )
