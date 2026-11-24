"""M4 Pro: compute each distinct causal token history once. Keep attention and readouts in the original batch."""
from importlib.metadata import version

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.llama import Model


def supported(model, quant):
    """The dispatch arithmetic below is private to these Metal kernels, not part of the MLX API."""
    if (mx.__version__ != "0.32.3" or not mx.metal.is_available() or mx.default_device() != mx.Device(mx.gpu)
            or mx.device_info().get("device_name") != "Apple M4 Pro" or version("mlx-lm") != "0.32.0"
            or type(model) is not Model or model.model.swa_idx is not None):
        return False
    a = model.args
    shape = a.num_hidden_layers, a.hidden_size, a.intermediate_size, a.num_attention_heads, a.num_key_value_heads
    if shape != (32, 1536, 8960, 12, 2) and not (quant and shape == (60, 2048, 11008, 16, 2)):
        return False
    # --quant q8 records no mode: nn.quantize's default, affine
    if a.head_dim != 128 or (quant and {"mode": "affine", **quant} != {"bits": 8, "group_size": 64, "mode": "affine"}):
        return False
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        if type(attn.rope) is not nn.RoPE or layer.input_layernorm.weight.dtype != mx.bfloat16:
            return False
        for p in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if quant:
                if (type(p) is not nn.QuantizedLinear or (p.bits, p.group_size, p.mode) != (8, 64, "affine")
                        or p.scales.dtype != mx.bfloat16):
                    return False
            elif type(p) is not nn.Linear or p.weight.dtype != mx.bfloat16:
                return False
    return True


def _tiles(n, width):
    return (n + width - 1) // width


def _splits(rows, k, n, group):
    # MLX v0.32.3 mlx/backend/metal/{quantized,matmul}.cpp: qmm_splitk and steel_gemm_splitk_axpby.
    if group:
        align = max(32, group)
        parts = min(max(1, 512 // (_tiles(rows, 32) * _tiles(n, 32))), k // align)
        while parts > 1 and k % (parts * align):
            parts -= 1
        return parts
    if k >= max(rows, n) and k >= 128 and _tiles(rows, 16) * _tiles(n, 16) <= 2048:
        ratio = (k // 16) // (_tiles(rows, 32) * _tiles(n, 32))
        return min(32, max(2, 1 << (max(1, ratio) - 1).bit_length()))
    return 1


def hidden(model, rows, prefix):
    """Last hidden state of each row, or None when compaction saves no work or needs a vector kernel."""
    batch, length = len(rows), max(map(len, rows))
    ref = batch * length
    ids, at, own, last, nodes = [], [], [], [], {}
    for b, row in enumerate(rows):
        parent = -1
        for j, token in enumerate(row):
            key = parent, token
            if key not in nodes:
                nodes[key] = len(ids)
                own.append(b * length + j)
                ids.append(token)
            parent = nodes[key]
            at.append(parent)
        last.append(parent)
        at.extend([parent] * (length - len(row)))
    count = len(ids)
    if count < 33 or count == ref:
        return None
    at, own = mx.array(at).reshape(batch, length), mx.array(own)
    offset = prefix[0][0].shape[2] if prefix else 0
    h = model.embed_tokens(mx.array([ids]))
    actions, zero = {}, mx.array([0])

    def project(layer, x):
        sig = x.shape[-1], layer.weight.shape[0], getattr(layer, "group_size", 0)
        if sig not in actions:
            source, target = _splits(ref, *sig), _splits(count, *sig)
            actions[sig] = "plain" if source == target else "gather" if sig[2] and source == 1 else "pad"
        if actions[sig] == "plain":
            return layer(x)
        if actions[sig] == "gather":  # one weight matrix, without the compact shape's extra split-K reduction
            y = mx.gather_qmm(x, layer.weight, layer.scales, layer.get("biases"), rhs_indices=zero,
                              bits=layer.bits, group_size=layer.group_size, mode=layer.mode)
            return y + layer.bias if "bias" in layer else y
        return layer(mx.pad(x, ((0, 0), (0, ref - count), (0, 0))))[:, :count]

    for i, layer in enumerate(model.layers):
        a = layer.self_attn
        x = layer.input_layernorm(h)

        def heads(proj, n):
            return project(proj, x)[0][at].reshape(batch, length, n, -1).transpose(0, 2, 1, 3)

        q = a.rope(heads(a.q_proj, a.n_heads), offset=offset)
        k = a.rope(heads(a.k_proj, a.n_kv_heads), offset=offset)
        v = heads(a.v_proj, a.n_kv_heads)
        if offset:
            pk, pv = prefix[i]
            k = mx.concatenate([mx.broadcast_to(pk, (batch, *pk.shape[1:])), k], axis=2)
            v = mx.concatenate([mx.broadcast_to(pv, (batch, *pv.shape[1:])), v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask="causal" if length > 1 else None)
        h = h + project(a.o_proj, o.transpose(0, 2, 1, 3).reshape(ref, -1)[own][None])
        x = layer.post_attention_layernorm(h)
        mlp = layer.mlp
        h = h + project(mlp.down_proj, swiglu(project(mlp.gate_proj, x), project(mlp.up_proj, x)))
        if i % 8 == 0:
            mx.async_eval(h)
    return model.norm(h[0][mx.array(last)])
