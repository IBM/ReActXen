from datetime import datetime


def get_chat_message(messages, is_system_prompt=False, replace_system_by_assistant=False):
    c_messages = []
    if isinstance(messages, str):  # Handle the autoregressive nature
        c_messages.append({"content": messages, "role": "user"})
    elif isinstance(messages, list) and len(messages) == 1:
        c_messages.append({"content": messages[0], "role": "user"})
    elif isinstance(messages, list) and is_system_prompt:
        if replace_system_by_assistant:
            c_messages.append({"content": messages[0], "role": "assistant"})
        else:
            c_messages.append({"content": messages[0], "role": "system"})
        if len(messages) > 1:
            c_messages.append({"content": messages[1], "role": "user"})
            for i in range(2, len(messages), 2):
                c_messages.append({"content": messages[i], "role": "assistant"})
                c_messages.append({"content": messages[i + 1], "role": "user"})
    elif isinstance(messages, list):
        c_messages.append({"content": messages[0], "role": "user"})
        for i in range(1, len(messages), 2):
            c_messages.append({"content": messages[i], "role": "assistant"})
            c_messages.append({"content": messages[i + 1], "role": "user"})
    else:
        pass
    return c_messages


def get_decorated_chat_template(user_message, trigger_statement=""):
    current_date = datetime.now()
    formatted_date = current_date.strftime("%B %d, %Y")
    return f"""<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: {formatted_date}.\nYou are Granite, developed by IBM. You are a helpful AI assistant<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>{user_message}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>{trigger_statement}\n\n"""
