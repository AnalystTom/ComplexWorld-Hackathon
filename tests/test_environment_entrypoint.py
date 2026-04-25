from pathlib import Path


def test_environment_entrypoint_hosts_network_benchmark_first() -> None:
    source = Path("environment.py").read_text()

    assert "Server(get_hosted_environment_classes()).run(port=port)" in source
    assert "return [NetworkBenchmarkEnv, HackerEnv]" in source
