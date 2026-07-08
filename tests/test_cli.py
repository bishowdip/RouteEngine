"""Smoke tests for the CLI front-end (synthetic graph keeps them offline/fast)."""

import pytest

from src.cli import build_parser, cmd_info, cmd_route


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_parser_route_arguments():
    parser = build_parser()
    args = parser.parse_args(["route", "--source", "0", "--target", "5",
                              "--alternatives", "2"])
    assert args.command == "route"
    assert args.source == 0 and args.target == 5 and args.alternatives == 2


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cmd_route_runs_on_small_synthetic(capsys, monkeypatch):
    # force the synthetic graph so the test never touches the network
    import src.cli as cli
    from src.data.loader import synthetic_city_graph
    monkeypatch.setattr(cli, "load_city_graph",
                        lambda max_nodes=None: (synthetic_city_graph(60, seed=1), "synthetic"))
    rc = cmd_route(_Args(limit=60, source=0, target=30, alternatives=2))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Dijkstra:" in out and "A*:" in out


def test_cmd_info_runs(capsys, monkeypatch):
    import src.cli as cli
    from src.data.loader import synthetic_city_graph
    monkeypatch.setattr(cli, "load_city_graph",
                        lambda max_nodes=None: (synthetic_city_graph(40, seed=2), "synthetic"))
    cmd_info(_Args(limit=40))
    assert "nodes=" in capsys.readouterr().out
