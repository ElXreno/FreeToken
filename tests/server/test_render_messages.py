from freetoken.server.generation import render_messages


def test_render_messages_merges_leading_system_run():
    msgs = render_messages(
        [
            {"role": "system", "content": "A"},
            {"role": "system", "content": [{"type": "text", "text": "B"}]},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "later"},
        ]
    )
    assert [m["role"] for m in msgs] == ["system", "user", "system"]
    assert msgs[0]["content"] == "A\n\nB"
    assert msgs[2]["content"] == "later"


def test_render_messages_keeps_a_single_system_message():
    msgs = render_messages([{"role": "system", "content": "A"}, {"role": "user", "content": "hi"}])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "A"


def test_render_messages_merges_system_run_with_image_parts():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = render_messages(
        [
            {"role": "system", "content": "A"},
            {"role": "system", "content": [{"type": "text", "text": "B"}, image]},
            {"role": "user", "content": "hi"},
        ]
    )
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"][0] == {"type": "text", "text": "A"}
    assert msgs[0]["content"][1] == {"type": "text", "text": "B"}
    assert msgs[0]["content"][2]["type"] == "image"
