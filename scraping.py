import random

from bs4 import BeautifulSoup
from requests import get


def get_useragent():
    return random.choice(_useragent_list)


# taken from <https://github.com/Nv7-GitHub/googlesearch/blob/master/googlesearch/user_agents.py>
_useragent_list = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.62",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0",
]


def _req(term, proxies, timeout):
    resp = get(
        url="https://www.google.com/search",
        headers={"User-Agent": get_useragent()},
        params={
            "q": term,
            "tbm": "isch",
        },
        cookies={"CONSENT": "YES+"},  # https://stackoverflow.com/a/70560365
        proxies=proxies,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp


import base64
import re
import urllib


def image_search(term, proxies=None, timeout=5):
    escaped_term = urllib.parse.quote_plus(term)
    # Proxy
    proxies = None
    # Fetch
    results = []
    resp = _req(escaped_term, proxies, timeout)
    soup = BeautifulSoup(resp.text, "html.parser")

    # for some reason, images are stored in a script tag
    # find all scripts starting with _setImgSrc(
    imgs = {}
    scripts = soup.find_all("script")
    for script in scripts:
        if script.string is None:
            continue
        script = script.string
        if not script.startswith("_setImgSrc("):
            continue
        match = re.search(r"_setImgSrc\('(i\d+)','(.*)'\);", script)
        assert match
        imgs[match.group(1)] = match.group(2)

    result_block = soup.find_all("img", attrs={"class": "rg_i"})
    for result in result_block:
        if result.has_attr("data-iid"):
            img = imgs[result.get("data-iid")]
        elif result.has_attr("src"):
            img = result.get("src")
        else:
            img = result.get("data-src")
        result = result.parent.parent.nextSibling
        title = result.get("title")
        href = result.get("href")
        results.append((href, title, img))

    return results


if __name__ == "__main__":
    import json

    from tqdm import tqdm

    with open("flavors.txt") as f:
        flavors = f.read().splitlines()
    flavors = dict([line.split(" - ") for line in flavors])
    with open("flavors.json", "w") as f:
        json.dump(flavors, f)
    flavors = list(flavors)
    all_results = {}
    for flavor in tqdm(flavors):
        results = image_search(flavor + " ice cream")[:10]

        log_results = []

        for i, (href, title, img) in enumerate(results):
            assert img.startswith("data:image\\")
            format = re.match(r"data:image\\/(.+);base64", img).group(1)
            assert format == "jpeg"
            img = base64.b64decode(img.split(",")[1] + "==")
            filename = f"{flavor} {i+1}.jpeg"
            log_results.append((filename, href, title))
            with open("imgs/" + filename, "wb") as f:
                f.write(img)
        all_results[flavor] = log_results

    with open("results.json", "w") as f:
        json.dump(all_results, f)

    with open("flavors.txt") as f:
        flavors = f.read().splitlines()
