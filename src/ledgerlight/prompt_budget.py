import re


def token_count(client, url, content):
    response = client.post(
        url + "/tokenize", json={"content": content, "add_special": False, "parse_special": True}
    )
    response.raise_for_status()
    return len(response.json()["tokens"])


def trim_tokens(client, url, text, limit):
    ends = [0] + [match.end() for match in re.finditer(r"\S+", text)]
    low, high = 0, len(ends) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if token_count(client, url, text[: ends[middle]]) <= limit:
            low = middle
        else:
            high = middle - 1
    result = text[: ends[low]]
    if token_count(client, url, result) > limit:
        raise ValueError("Token trimming exceeded the requested limit")
    return result


def pack_prompt(client, url, system, question, issued, cfg, thinking):
    entries = []
    for marker, item in issued.items():
        sources = "; ".join(
            f"{s['ticker']} / {s['form']} / {s['year']} / page {s['page']}" for s in item["sources"]
        )
        entries.append([marker + " " + sources, "\n".join(item["windows"])])
    clipped = False
    while True:
        user = (
            "Question: "
            + question
            + "\n\nSource excerpts:\n\n"
            + "\n\n".join(header + "\n" + text for header, text in entries)
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = client.post(
            url + "/apply-template",
            json={"messages": messages, "chat_template_kwargs": {"enable_thinking": thinking}},
        )
        response.raise_for_status()
        count = token_count(client, url, response.json()["prompt"])
        overflow = count + cfg["max_tokens"] + 256 - cfg["context"]
        if overflow <= 0:
            return messages, count, clipped
        lengths = [token_count(client, url, text) for _, text in entries]
        if not lengths or not max(lengths):
            raise ValueError("Question and metadata exceed the local context budget")
        chosen = max(range(len(entries)), key=lambda index: (lengths[index], -index))
        entries[chosen][1] = trim_tokens(
            client, url, entries[chosen][1], max(0, lengths[chosen] - max(1, overflow))
        )
        clipped = True
