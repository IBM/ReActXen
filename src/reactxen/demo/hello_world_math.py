import argparse
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
import json


def main(mode, model_id):
    # Define the question
    question = "Find the $r$ that satisfies $\\log_{16} (r+16) = \\frac{5}{4}$."

    # Set the agent parameters based on the mode (either 'text' or 'code')
    if mode == "code":
        actionstyle = "Code"
    elif mode == "text":
        actionstyle = "Text"
    else:
        raise ValueError("Invalid mode. Please choose either 'text' or 'code'.")

    kwargs = {
        "actionstyle": actionstyle,  # Using the specified action style
        "max_steps": 6,  # Limit the number of steps
        "num_reflect_iteration": 3,  # Example number of iterations for reflection
        "early_stop": False,  # Do not stop early
        "debug": False,
        "reactstyle": "thought_and_act_together"
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
    with open("sample_review_math_problem.json", "w") as file:
        json.dump(ans, file, indent=4)

    # print(ans)
    metric = agent.export_benchmark_metric()
    print (metric)
    with open("sample_metric_math_problem.json", "w") as file:
        json.dump(metric, file, indent=4)

    # explort trajectroy
    traj = agent.export_trajectory()
    with open("sample_traj_math_problem.json", "w") as file:
        json.dump(traj, file, indent=4)

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
