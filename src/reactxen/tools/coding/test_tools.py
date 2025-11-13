import json
import traceback
from code_generator_tool import CodeGeneratorTool
from sandbox_executor_tool import SandboxExecutorTool


def test_code_generator_basic():
    """Test that the code generator produces valid Python code."""
    tool = CodeGeneratorTool()
    query = "Compute the first 5 square numbers and print them as JSON under key 'result'."

    code = tool.run(query)
    print("\n[Generated Code]\n", code)

    assert isinstance(code, str), "Generated output is not a string."
    assert "print" in code, "Generated code missing print statement."
    assert "json" in code, "Generated code missing JSON handling."


def test_sandbox_executor_basic():
    """Test sandbox execution of a simple, known-safe Python snippet."""
    code = """
import json
squares = [x * x for x in range(5)]
print(json.dumps({"result": squares}))
"""

    executor = SandboxExecutorTool()
    result = executor.run(code)
    print("\n[Sandbox Execution Output]\n", result)

    assert isinstance(result, dict), "Executor did not return a dictionary."
    assert result.get("status") == "success", f"Execution failed: {result}"
    assert '"result"' in result.get("stdout", ""), "Output missing expected JSON key."


def test_generator_and_executor_integration():
    """End-to-end test: LLM-generated code is executed in sandbox and produces correct output."""
    generator = CodeGeneratorTool()
    executor = SandboxExecutorTool()

    query = "Find the factorial of 5 and print it as JSON under key 'result'."

    try:
        # Generate Python code from natural language
        code = generator.run(query)
        print("\n[Generated Code]\n", code)

        # Execute the generated code safely
        result = executor.run(code)
        print("\n[Execution Result]\n", result)

        # Check execution success
        assert result.get("status") == "success", f"Execution failed: {result}"

        # Parse and verify output correctness
        output = json.loads(result["stdout"])
        assert output["result"] == 120, f"Unexpected factorial result: {output}"

    except Exception as e:
        traceback.print_exc()
        raise AssertionError(f"Integration test failed: {e}")


if __name__ == "__main__":
    print("=== 🧪 Running Tool Tests ===")
    test_code_generator_basic()
    test_sandbox_executor_basic()
    test_generator_and_executor_integration()
    print("\n✅ All tests passed successfully!")
