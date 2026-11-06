import json
import threading


def test_status_reader_never_observes_partial_json(tmp_path):
    from unisupermimic.deploy.run_sim2sim import write_json

    path = tmp_path / "status.json"
    write_json(path, {"values": []})
    stop, errors = threading.Event(), []

    def read():
        while not stop.is_set():
            try:
                json.loads(path.read_text())
            except (ValueError, OSError) as exc:
                errors.append(str(exc))

    thread = threading.Thread(target=read)
    thread.start()
    try:
        for count in range(50):
            write_json(path, {"values": list(range(500)), "count": count})
    finally:
        stop.set()
        thread.join()
    assert not errors
    assert json.loads(path.read_text())["count"] == 49
