import argparse
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
import json
from reactxen.tools.iot.bmstool import BMSSensors
from reactxen.agents.react.prompts.fewshots import MPE_SIMPLE4
from reactxen.utils.tool_description import get_tool_description

def main(mode, model_id):
    # Define the question
    question = "Retrieve metadata for Chiller 6 located at the MAIN site."

    # Set the agent parameters based on the mode (either 'text' or 'code')
    if mode == "code":
        actionstyle = "Code"
    elif mode == "text":
        actionstyle = "Text"
    else:
        raise ValueError("Invalid mode. Please choose either 'text' or 'code'.")

    tools = [
        BMSSensors(),
    ]

    kwargs = {
        "actionstyle": actionstyle,  # Using the specified action style
        "max_steps": 6,  # Limit the number of steps
        "num_reflect_iteration": 3,  # Example number of iterations for reflection
        "early_stop": False,  # Do not stop early
        "debug": False,
        "reactstyle": "thought_and_act_together",
        "tools": tools,      
        "tool_desc": get_tool_description(tools),
        "tool_names": [ 'sensors' ],
        "react_example": MPE_SIMPLE4 + """
            Question: download metadata for Chiller 4 at MAIN site
            Thought 1: I need to get the sensor data for Chiller 4 at site MAIN to answer the question.
            Action 1: sensors
            Action Input 1: assetnum=Chiller 4, site_name=MAIN
            Observation 1: {"site_name": "MAIN", "assetnum": "Chiller 4", "total_sensors": 2, "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c6571941-4857-4701-bd8a-9a28fa2435c3.json", "message": "found 2 sensors for assetnum Chiller 4 and site_name MAIN. file_path contains a JSON array of Sensor data"}
            Thought 2: I now have the sensor data for Chiller 4 at site MAIN, which is stored in a file.
            Action 2: Finish
            Action Input 2: The sensor data for Chiller 4 at site MAIN has been downloaded and is listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c6571941-4857-4701-bd8a-9a28fa2435c3.json.
        """,
    }

    # Call create_reactxen_agent to initialize the agent
    agent = create_reactxen_agent(
        question=question,
        key="",  # Provide any necessary key if required
        react_llm_model_id=model_id,  # Pass the model_id
        reflect_llm_model_id=model_id,
        **kwargs  # Pass all additional parameters
    )

    # Run the agent and output the result
    ans = agent.run()

    # Print the result
    print (ans)
    # with open("sample_review_math_problem.json", "w") as file:
    #     json.dump(ans, file, indent=4)

    # print(ans)
    metric = agent.export_benchmark_metric()
    print (metric)
    # with open("sample_metric_math_problem.json", "w") as file:
    #     json.dump(metric, file, indent=4)

    # explort trajectroy
    traj = agent.export_trajectory()
    # with open("sample_traj_math_problem.json", "w") as file:
    #     json.dump(traj, file, indent=4)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run the ReActXen agent with specified mode and model_id."
    )
    parser.add_argument(
        "--mode",
        choices=["text", "code"],
        default="code",
        help="Specify the mode: 'text' or 'code'.",
    )
    parser.add_argument(
        "--model_id", type=int, default=8, help="Specify the model_id (default is 8)."
    )

    # Get the arguments from the command line
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(mode=args.mode, model_id=args.model_id)
