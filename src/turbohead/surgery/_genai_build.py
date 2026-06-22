"""Run the onnxruntime-genai model builder with a shim for the transformers<->genai Gemma3 rope
mismatch, then delegate to the real builder CLI (same args).

genai 0.14.1's Gemma builder reads flat `config.rope_local_base_freq` / `config.rope_theta`;
transformers 5.12 moved those into `config.rope_parameters` ({sliding_attention, full_attention}),
so the builder AttributeErrors. We re-expose the flats on the loaded config. No-op for every model
that already has them (Qwen etc.).

ponytail: narrow compat shim, not a fork — delete when genai ships a builder that reads
`rope_parameters`. Usage is identical to `python -m onnxruntime_genai.models.builder ...`.
"""
import runpy
import transformers

_orig = transformers.AutoConfig.from_pretrained


def _patched(*args, **kwargs):
    cfg = _orig(*args, **kwargs)
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    for c in {id(cfg): cfg, id(text): text}.values():  # dedupe; configs aren't hashable
        rp = getattr(c, "rope_parameters", None)
        if rp and not hasattr(c, "rope_local_base_freq"):
            c.rope_local_base_freq = rp.get("sliding_attention", {}).get("rope_theta", 10000.0)
            c.rope_theta = rp.get("full_attention", {}).get("rope_theta", getattr(c, "rope_theta", 1e6))
    return cfg


transformers.AutoConfig.from_pretrained = staticmethod(_patched)
runpy.run_module("onnxruntime_genai.models.builder", run_name="__main__")  # parses sys.argv as usual
