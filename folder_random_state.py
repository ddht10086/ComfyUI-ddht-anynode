# SPDX-License-Identifier: GPL-3.0-only
"""Persistent, per-folder recency weights. No ComfyUI or tensor dependencies."""

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import time
import uuid


SCHEMA_VERSION = 1
FINGERPRINT_CACHE_VERSION = 1
ALGORITHM = "successful_draw_age_linear_v1"
LOGGER = logging.getLogger("DDHT.FolderImages")


def _now():
    return datetime.now(timezone.utc).isoformat()


def canonical_folder(folder):
    return os.path.normcase(str(Path(folder).resolve()))


def state_path_for(folder, state_directory):
    key = hashlib.sha256(canonical_folder(folder).encode("utf-8")).hexdigest()
    return Path(state_directory).resolve() / f"{key}.json"


def _stat_key(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def fingerprint_file(path, check_interrupt=lambda: None):
    """Read bytes every time, including same-size replacements with restored mtime."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            before = os.fstat(file.fileno())
            while True:
                check_interrupt()
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(file.fileno())
            check_interrupt()
            # Keep both snapshots on the same API: Windows stat(path) may use
            # creation time for ctime while fstat(fd) uses metadata change time.
            # Reopen the path while the first handle is alive to also detect a
            # file replaced at that path, including same-size/mtime replacements.
            with path.open("rb") as current:
                current_stat = os.fstat(current.fileno())
            if _stat_key(before) != _stat_key(after) or _stat_key(after) != _stat_key(current_stat):
                raise RuntimeError(f"图片在扫描期间发生变化，请重新执行：{path.name}")
        return {"size": after.st_size, "sha256": digest.hexdigest()}
    except OSError as error:
        raise RuntimeError(f"无法读取图片内容指纹，请检查文件及权限：{path.name}") from error


def _path_signature(path):
    # Only compare these values with other path.stat() snapshots, never fstat().
    try:
        return list(_stat_key(path.stat()))
    except OSError as error:
        raise RuntimeError(f"无法读取图片文件属性，请检查文件及权限：{path.name}") from error


def _cached_images(state, folder, files, recursive, reset_marker):
    """Trust a cache only for a valid document with exactly the same file list."""
    if not isinstance(state, dict) or state.get("fingerprint_cache_version") != FINGERPRINT_CACHE_VERSION:
        return {}
    images = state.get("images")
    if not isinstance(images, list) or len(images) != len(files):
        return {}  # Any count change must rehash every remaining/new image.
    cached = {}
    for item in images:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            return {}
        name, signature, digest = item["relative_path"], item.get("stat_signature"), item.get("sha256")
        if (name in cached or not _integer(item.get("size"))
                or not isinstance(signature, list) or len(signature) != 5
                or any(type(value) is not int for value in signature) or signature[2] != item["size"]
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            return {}
        cached[name] = item
    names = [path.relative_to(folder).as_posix() for path in files]
    if set(names) != set(cached):
        return {}  # Also handle rename and remove+add with unchanged count.
    manifest = [{key: cached[name][key] for key in ("relative_path", "size", "sha256")} for name in names]
    if _rebuild_reason(state, folder, recursive, manifest, reset_marker) is not None:
        return {}
    return cached


def build_manifest(folder, files, check_interrupt, cached=None):
    manifest = []
    for path in files:
        check_interrupt()
        name = path.relative_to(folder).as_posix()
        signature = _path_signature(path)
        previous = (cached or {}).get(name)
        if previous is not None and previous["stat_signature"] == signature:
            fingerprint = {key: previous[key] for key in ("size", "sha256")}
        else:
            fingerprint = fingerprint_file(path, check_interrupt)
            if signature != _path_signature(path):
                raise RuntimeError(f"图片在扫描期间发生变化，请重新执行：{path.name}")
        manifest.append({"relative_path": name, **fingerprint, "stat_signature": signature})
    return manifest


def _manifest_digest(manifest):
    # Metadata-only changes refresh the cache without resetting usage weights.
    # Keep the original digest format so old documents retain their history.
    content = [{key: item[key] for key in ("relative_path", "size", "sha256")} for item in manifest]
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _document_lock(path, check_interrupt, timeout=30.0):
    """OS locks coordinate threads, node instances and ComfyUI processes locally."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file = path.with_suffix(".json.lock").open("a+b")
    except OSError as error:
        raise RuntimeError(f"无法打开权重文档目录，请检查权限：{path.parent}") from error
    acquired = False
    try:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"\0")
            file.flush()
        deadline = time.monotonic() + timeout
        while not acquired:
            check_interrupt()
            try:
                file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError("该文件夹的权重文档正在被其他任务使用，请稍后重试。") from None
                time.sleep(0.05)
        yield
    finally:
        try:
            if acquired:
                file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        finally:
            file.close()


def _write_document(path, state):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem + "_", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(state, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"无法保存权重文档，未完成本次图片输出：{path}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _rank_images(state):
    for item in state["images"]:
        item["weight"] = state["total_draws"] - item["last_used_draw"] + 1
    state["images"].sort(key=lambda item: (-item["weight"], item["relative_path"]))
    for rank, item in enumerate(state["images"], 1):
        item["rank"] = rank


def _new_document(folder, recursive, manifest, reset_marker, reason):
    now = _now()
    state = {
        "schema_version": SCHEMA_VERSION, "algorithm": ALGORITHM,
        "fingerprint_cache_version": FINGERPRINT_CACHE_VERSION,
        "folder": canonical_folder(folder), "recursive": recursive,
        "generation_id": uuid.uuid4().hex, "rebuild_reason": reason,
        "created_at": now, "updated_at": now, "manifest_sha256": _manifest_digest(manifest),
        "reset_marker": reset_marker, "total_draws": 0, "image_count": len(manifest),
        "weight_rule": "weight = total_draws - last_used_draw + 1; never used: last_used_draw = 0",
        "images": [{**entry, "use_count": 0, "last_used_draw": 0, "last_used_at": None}
                   for entry in manifest],
    }
    _rank_images(state)
    return state


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _rebuild_reason(state, folder, recursive, manifest, reset_marker):
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION or state.get("algorithm") != ALGORITHM:
        return "invalid_or_old_document"
    if state.get("folder") != canonical_folder(folder) or state.get("recursive") is not recursive:
        return "folder_or_scan_changed"
    if state.get("manifest_sha256") != _manifest_digest(manifest):
        return "image_list_or_content_changed"
    if state.get("reset_marker") != reset_marker:
        return "manual_reset"
    total, images = state.get("total_draws"), state.get("images")
    if not _integer(total) or not isinstance(images, list) or len(images) != len(manifest) or state.get("image_count") != len(manifest):
        return "invalid_document"
    expected = {item["relative_path"]: item for item in manifest}
    seen, used_draws, use_count_sum = set(), set(), 0
    for item in images:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            return "invalid_document"
        name = item["relative_path"]
        count, last = item.get("use_count"), item.get("last_used_draw")
        if name in seen or name not in expected or not _integer(count) or not _integer(last):
            return "invalid_document"
        if count > last or last > total or (count == 0) != (last == 0) or (last and last in used_draws):
            return "invalid_document"
        if any(item.get(key) != expected[name][key] for key in ("sha256", "size")):
            return "invalid_document"
        if type(item.get("weight")) is not int or item["weight"] != total - last + 1:
            return "invalid_document"
        seen.add(name)
        used_draws.add(last)
        use_count_sum += count
    if use_count_sum != total or (total and total not in used_draws):
        return "invalid_document"
    return None


def _load_document(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeError):
        return None
    except OSError as error:
        raise RuntimeError(f"无法读取权重文档，请检查权限：{path}") from error


def _weighted_index(state, manifest, seed):
    by_name = {item["relative_path"]: item for item in state["images"]}
    weights = [by_name[item["relative_path"]]["weight"] for item in manifest]
    # Include the saved draw counter so fixed seed still progresses across restarts.
    rng = random.Random(f"DDHTWeighted:v1:{seed}:{state['total_draws']}")
    ticket = rng.randrange(sum(weights))
    for index, weight in enumerate(weights):
        if ticket < weight:
            return index
        ticket -= weight
    raise AssertionError("Weighted selection exhausted its probability mass")


def run_weighted(folder, files, recursive, seed, reset_marker, state_directory,
                 consume, check_interrupt=lambda: None):
    """Run consume(index), then atomically commit history; failures don't count as uses."""
    path = state_path_for(folder, state_directory)
    with _document_lock(path, check_interrupt):
        state = _load_document(path)
        cached = _cached_images(state, folder, files, recursive, reset_marker)
        manifest = build_manifest(folder, files, check_interrupt, cached)
        reason = "first_use" if state is None else _rebuild_reason(state, folder, recursive, manifest, reset_marker)
        if reason:
            state = _new_document(folder, recursive, manifest, reset_marker, reason)
            _write_document(path, state)
            LOGGER.info("加权文档已新建或重建（%s）：%s", reason, path)
        if not files:
            raise ValueError("文件夹中没有支持的图片；加权文档已重建为空列表。")
        index = _weighted_index(state, manifest, seed)
        check_interrupt()
        result = consume(index)
        check_interrupt()
        if fingerprint_file(files[index], check_interrupt) != {key: manifest[index][key] for key in ("size", "sha256")}:
            if _path_signature(files[index]) == manifest[index]["stat_signature"]:
                # Bytes changed while all cached attributes stayed identical.
                # Invalidate the cache so retrying can recover without a manual
                # reset; do not count the failed output as a successful use.
                state.pop("fingerprint_cache_version", None)
                _write_document(path, state)
            raise RuntimeError("所选图片与扫描记录不一致，可能在读取期间被修改，请重新执行；本次未更新使用权重。")
        state["total_draws"] += 1
        now = _now()
        selected = next(item for item in state["images"] if item["relative_path"] == manifest[index]["relative_path"])
        selected["use_count"] += 1
        selected["last_used_draw"] = state["total_draws"]
        selected["last_used_at"] = now
        state["updated_at"] = now
        state["last_seed"] = seed
        signatures = {item["relative_path"]: item["stat_signature"] for item in manifest}
        for item in state["images"]:
            item["stat_signature"] = signatures[item["relative_path"]]
        state["fingerprint_cache_version"] = FINGERPRINT_CACHE_VERSION
        _rank_images(state)
        _write_document(path, state)
        return result, str(path)
