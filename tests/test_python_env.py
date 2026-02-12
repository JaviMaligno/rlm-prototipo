from rlm.python_env import PythonEnv


def test_python_env_exec_and_helpers():
    env = PythonEnv()
    env.set_context("hello world")
    result = env.exec("print(get_slice(0, 5))")
    assert result["ok"] is True
    assert "hello" in result["stdout"]

    result2 = env.exec("print(search('world', max_results=1))")
    assert result2["ok"] is True
    assert "world" in result2["stdout"]
