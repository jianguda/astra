from pathlib import Path
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from .nethook import Trace
from .runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def _context_length(model) -> int:
    """Context length of the text decoder, capped at 4096 tokens to keep the statistics pass in memory."""
    configs = [model.config, getattr(model.config, 'text_config', None)]
    for config in configs:
        for attr in ('n_positions', 'max_sequence_length', 'max_position_embeddings', 'seq_length'):
            value = getattr(config, attr, None) if config is not None else None
            if isinstance(value, int) and value > 0:
                return min(value, 4096)
    raise NotImplementedError


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """Load or compute the cached statistics of the inputs of `layer_name` over `ds_name`."""

    def get_ds():
        # parquet releases on the Hub (script-based dataset repositories are no longer loadable)
        repository, config = {
            "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            "wikipedia": ("wikimedia/wikipedia", "20231101.en"),
        }[ds_name]
        raw_ds = load_dataset(repository, config)
        maxlen = _context_length(model)
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 100  # kept small: sequences are long
    npos = _context_length(model)

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    ds = get_ds() if not filename.exists() else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat
