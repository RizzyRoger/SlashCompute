from slashcompute.pipeline.model_profile import profile_from


def _t(nbytes):
    return {"dtype": "BF16", "shape": [nbytes // 2], "data_offsets": [0, nbytes]}


def test_multimodal_counts_only_text_layers_and_prefers_text_config():
    # Gemma3-style: 4 text layers, 6 vision layers, text dims under text_config
    headers = {"language_model.model.embed_tokens.weight": _t(1000),
               "language_model.model.norm.weight": _t(10),
               "multi_modal_projector.mm_input_projection_weight": _t(500),
               "vision_tower.vision_model.embeddings.patch_embedding.weight": _t(700)}
    for i in range(4):
        headers[f"language_model.model.layers.{i}.mlp.up_proj.weight"] = _t(100 + i)
    for i in range(6):
        headers[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = _t(5000)
    config = {"model_type": "gemma3", "hidden_size": 1152, "num_hidden_layers": 6,
              "text_config": {"hidden_size": 640, "num_hidden_layers": 4, "vocab_size": 1000,
                              "num_attention_heads": 4, "num_key_value_heads": 1,
                              "head_dim": 256, "intermediate_size": 2048},
              "tie_word_embeddings": True}

    p = profile_from("gemma3", config, headers)

    assert p.num_layers == 4
    assert p.hidden_size == 640
    assert p.vocab_size == 1000
    assert p.layer_bytes == (100, 101, 102, 103)
    assert p.embed_bytes == 1000
    assert p.head_bytes == 10 + 1000  # final norm + tied embedding, no vision weights
    assert p.head_params == 640 * 1000
