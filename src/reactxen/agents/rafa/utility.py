import re
import json 

def generate_future_thinkact(current_scratchpad, bandwidth):
    """
    Generates multiple future think-act plans based on the current scratchpad.

    Args:
    - current_scratchpad (dict): The current state or trajectory (could be a dictionary or other object).
    - bandwidth (int): The number of future plans to generate at this step.

    Returns:
    - plans (list): List of generated future plans based on the current scratchpad.
    """
    plans = []
    for i in range(bandwidth):
        # Here we simply replicate the scratchpad for now, 
        # but this can be expanded to generate different plans using `current_scratchpad`.
        new_plan = current_scratchpad  # In a real scenario, modify the state based on the plan
        plans.append(new_plan)
    return plans

def generate_future_observation(current_scratchpad):
    """
    Generates an observation from the current scratchpad.
    For now, it simply returns the scratchpad as is, but this can include observations 
    based on environment feedback.

    Args:
    - current_scratchpad (dict): The current state or trajectory.

    Returns:
    - observation (dict): The current state as an observation.
    """
    return current_scratchpad  # Placeholder for future observation logic

def generate_best_action(current_scratchpad, bandwidth, depth):
    """
    Generates the best action by exploring future plans at each step, considering the given depth.

    Args:
    - current_scratchpad (dict): The current state or trajectory.
    - bandwidth (int): The number of candidate plans to explore at each step.
    - depth (int): The depth of planning (how many steps into the future to consider).

    Returns:
    - best_action (dict): The best action from the final evaluated plans.
    """
    plans = [current_scratchpad]  # Start with the initial state (current scratchpad)
    
    # Iteratively generate future plans
    for i in range(depth):
        plan_in_step = []  # List of plans generated in the current step
        
        for p in plans:
            # Generate future think-act sequences for each plan
            cc = generate_future_thinkact(p, bandwidth)
            plan_in_step.extend(cc)  # Add new plans to the current step
        
        # After considering all plans at the current depth, update the plan list
        plans.extend(plan_in_step)  # Add all newly generated plans to the main plan list
    
    # Evaluate the plans (In this case, we just return the first plan, but you can add a selection criterion)
    # For simplicity, let's assume the first plan is the best (you can expand this with evaluation logic)
    best_plan = plans[0]  # Placeholder for a better evaluation mechanism
    
    # Assuming the best action is derived from the best plan (you might want to select based on some criterion)
    best_action = best_plan  # You can further refine this selection based on your problem specifics
    
    return best_action


def extract_and_parse_json(response):
    """
    Extract and parse JSON from the response.

    Args:
        response (str): The raw response from the LLM.

    Returns:
        dict: Parsed JSON object or an error report.
    """
    try:
        # Extract JSON block enclosed in ```json ... ```
        # match = re.search(r"```json(.*?)```", response, re.DOTALL)
        match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        if match:
            json_block = match.group(0).strip()  # Extract and clean the JSON block
        else:
            json_block = response.strip()

        if not json_block:
            raise ValueError("Extracted JSON block is empty.")

        parsed_json = json.loads(json_block)
        # print (parsed_json)
        return parsed_json

    except json.JSONDecodeError as ex:
        # print(f'came here : {ex}')
        # print(f'{response}')
        # Return error information if parsing fails
        return {
            "status": "Error",
            "value_function_score": 0
        }

    except ValueError as ex:
        # print(f"Value Error: {ex}")
        return {
            "status": "Error",
            "value_function_score": 0
        }
