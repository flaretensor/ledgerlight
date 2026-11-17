import os
import shutil
import urllib.request


def download(url, target, verify=None):
    part = target.with_name(target.name + ".part")
    part.unlink(missing_ok=True)
    if target.is_file():
        if verify is None:
            return
        try:
            verify(target)
            return
        except ValueError:
            pass
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if verify is not None:
            verify(part)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
