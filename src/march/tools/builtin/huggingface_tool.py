"""Hugging Face tool — search models/datasets, download files, get model info."""

from __future__ import annotations

import asyncio
from functools import partial

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.huggingface")

_MAX_OUTPUT = 8000


def _sync_hf_call(action: str, **kwargs) -> str:
    """Run HuggingFace Hub API calls synchronously (called in executor)."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return "Error: huggingface_hub not installed. Run: pip install huggingface_hub"

    api = HfApi()

    if action == "search_models":
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", "downloads")
        task = kwargs.get("task", "")

        filter_kwargs = {}
        if task:
            filter_kwargs["task"] = task

        models = list(api.list_models(
            search=query, sort=sort, limit=limit, **filter_kwargs
        ))

        if not models:
            return f"No models found for: {query}"

        lines = [f"Models matching '{query}' (sorted by {sort}):\n"]
        for m in models:
            downloads = getattr(m, "downloads", 0) or 0
            likes = getattr(m, "likes", 0) or 0
            pipeline = getattr(m, "pipeline_tag", "") or ""
            lines.append(f"  {m.id}")
            lines.append(f"    ↓{downloads:,}  ♥{likes}  {pipeline}")
        return "\n".join(lines)

    elif action == "search_datasets":
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", "downloads")

        datasets = list(api.list_datasets(search=query, sort=sort, limit=limit))

        if not datasets:
            return f"No datasets found for: {query}"

        lines = [f"Datasets matching '{query}':\n"]
        for d in datasets:
            downloads = getattr(d, "downloads", 0) or 0
            likes = getattr(d, "likes", 0) or 0
            lines.append(f"  {d.id}")
            lines.append(f"    ↓{downloads:,}  ♥{likes}")
        return "\n".join(lines)

    elif action == "model_info":
        model_id = kwargs.get("model_id", "")
        if not model_id:
            return "Error: model_id required"

        try:
            info = api.model_info(model_id)
        except Exception as e:
            return f"Error: {e}"

        tags = ", ".join(getattr(info, "tags", [])[:10]) or "none"
        siblings = getattr(info, "siblings", []) or []
        files = [s.rfilename for s in siblings[:20]]

        lines = [
            f"Model: {info.id}",
            f"Pipeline: {getattr(info, 'pipeline_tag', 'N/A')}",
            f"Downloads: {getattr(info, 'downloads', 0):,}",
            f"Likes: {getattr(info, 'likes', 0)}",
            f"Tags: {tags}",
            f"Library: {getattr(info, 'library_name', 'N/A')}",
            f"License: {getattr(info, 'card_data', {}) and getattr(info.card_data, 'license', 'N/A') or 'N/A'}",
            f"Files ({len(siblings)} total): {', '.join(files)}",
        ]
        return "\n".join(lines)

    elif action == "dataset_info":
        dataset_id = kwargs.get("dataset_id", "")
        if not dataset_id:
            return "Error: dataset_id required"

        try:
            info = api.dataset_info(dataset_id)
        except Exception as e:
            return f"Error: {e}"

        tags = ", ".join(getattr(info, "tags", [])[:10]) or "none"

        lines = [
            f"Dataset: {info.id}",
            f"Downloads: {getattr(info, 'downloads', 0):,}",
            f"Likes: {getattr(info, 'likes', 0)}",
            f"Tags: {tags}",
        ]
        return "\n".join(lines)

    elif action == "download":
        repo_id = kwargs.get("repo_id", "")
        filename = kwargs.get("filename", "")
        if not repo_id or not filename:
            return "Error: repo_id and filename required"

        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            return f"Downloaded to: {path}"
        except Exception as e:
            return f"Error downloading: {e}"

    else:
        return f"Error: Unknown action '{action}'. Use: search_models, search_datasets, model_info, dataset_info, download"


@tool(
    name="huggingface",
    description=(
        "Search and explore Hugging Face models and datasets. "
        "Actions: search_models, search_datasets, model_info, dataset_info, download."
    ),
)
async def huggingface_tool(
    action: str,
    query: str = "",
    model_id: str = "",
    dataset_id: str = "",
    repo_id: str = "",
    filename: str = "",
    task: str = "",
    sort: str = "downloads",
    limit: int = 10,
) -> str:
    """Interact with Hugging Face Hub.

    Args:
        action: One of: search_models, search_datasets, model_info, dataset_info, download.
        query: Search query (for search_models, search_datasets).
        model_id: Model ID like 'meta-llama/Llama-3.1-8B' (for model_info).
        dataset_id: Dataset ID like 'squad' (for dataset_info).
        repo_id: Repository ID (for download).
        filename: File to download from repo (for download).
        task: Filter by task (e.g. 'text-generation', 'image-classification').
        sort: Sort by: downloads, likes, trending, created, modified. Default: downloads.
        limit: Max results (1-50). Default: 10.

    Examples:
        action="search_models", query="llama 3", task="text-generation", limit=5
        action="model_info", model_id="meta-llama/Llama-3.1-8B"
        action="search_datasets", query="code generation"
        action="download", repo_id="bert-base-uncased", filename="config.json"
    """
    if not action.strip():
        return "Error: action required. Use: search_models, search_datasets, model_info, dataset_info, download"

    limit = max(1, min(50, limit))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            _sync_hf_call, action,
            query=query, model_id=model_id, dataset_id=dataset_id,
            repo_id=repo_id, filename=filename, task=task,
            sort=sort, limit=limit,
        ),
    )

    if len(result) > _MAX_OUTPUT:
        result = result[:_MAX_OUTPUT] + f"\n... (truncated)"

    return result
