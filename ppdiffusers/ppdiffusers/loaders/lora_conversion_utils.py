# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from ..utils import logging

logger = logging.get_logger(__name__)


def _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config, delimiter="_", block_slice_pos=5):
    # 1. get all state_dict_keys
    all_keys = list(state_dict.keys())
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

    # 2. check if needs remapping, if not return original dict
    is_in_sgm_format = False
    for key in all_keys:
        if any(p in key for p in sgm_patterns):
            is_in_sgm_format = True
            break

    if not is_in_sgm_format:
        return state_dict

    # 3. Else remap from SGM patterns
    new_state_dict = {}
    inner_block_map = ["resnets", "attentions", "upsamplers"]

    # Retrieves # of down, mid and up blocks
    input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()

    for layer in all_keys:
        if "text" in layer:
            new_state_dict[layer] = state_dict.pop(layer)
        else:
            layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
            if sgm_patterns[0] in layer:
                input_block_ids.add(layer_id)
            elif sgm_patterns[1] in layer:
                middle_block_ids.add(layer_id)
            elif sgm_patterns[2] in layer:
                output_block_ids.add(layer_id)
            else:
                raise ValueError(f"Checkpoint not supported because layer {layer} not supported.")

    input_blocks = {
        layer_id: [key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key]
        for layer_id in input_block_ids
    }
    middle_blocks = {
        layer_id: [key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key]
        for layer_id in middle_block_ids
    }
    output_blocks = {
        layer_id: [key for key in state_dict if f"output_blocks{delimiter}{layer_id}" in key]
        for layer_id in output_block_ids
    }

    # Rename keys accordingly
    for i in input_block_ids:
        block_id = (i - 1) // (unet_config.layers_per_block + 1)
        layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)

        for key in input_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = inner_block_map[inner_block_id] if "op" not in key else "downsamplers"
            inner_layers_in_block = str(layer_in_block_id) if "op" not in key else "0"
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + [str(block_id), inner_block_key, inner_layers_in_block]
                + key.split(delimiter)[block_slice_pos + 1 :]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in middle_block_ids:
        key_part = None
        if i == 0:
            key_part = [inner_block_map[0], "0"]
        elif i == 1:
            key_part = [inner_block_map[1], "0"]
        elif i == 2:
            key_part = [inner_block_map[0], "1"]
        else:
            raise ValueError(f"Invalid middle block id {i}.")

        for key in middle_blocks[i]:
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1] + key_part + key.split(delimiter)[block_slice_pos:]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in output_block_ids:
        block_id = i // (unet_config.layers_per_block + 1)
        layer_in_block_id = i % (unet_config.layers_per_block + 1)

        for key in output_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = inner_block_map[inner_block_id]
            inner_layers_in_block = str(layer_in_block_id) if inner_block_id < 2 else "0"
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + [str(block_id), inner_block_key, inner_layers_in_block]
                + key.split(delimiter)[block_slice_pos + 1 :]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    if len(state_dict) > 0:
        raise ValueError("At this point all state dict entries have to be converted.")

    return new_state_dict


def _convert_kohya_lora_to_diffusers(state_dict, unet_name="unet", text_encoder_name="text_encoder"):
    unet_state_dict = {}
    te_state_dict = {}
    te2_state_dict = {}
    network_alphas = {}

    # every down weight has a corresponding up weight and potentially an alpha weight
    lora_keys = [k for k in state_dict.keys() if k.endswith("lora_down.weight")]
    for key in lora_keys:
        lora_name = key.split(".")[0]
        lora_name_up = lora_name + ".lora_up.weight"
        lora_name_alpha = lora_name + ".alpha"

        if lora_name.startswith("lora_unet_"):
            diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

            if "input.blocks" in diffusers_name:
                diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
            else:
                diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")

            if "middle.block" in diffusers_name:
                diffusers_name = diffusers_name.replace("middle.block", "mid_block")
            else:
                diffusers_name = diffusers_name.replace("mid.block", "mid_block")
            if "output.blocks" in diffusers_name:
                diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
            else:
                diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")

            diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
            diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
            diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
            diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
            diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
            diffusers_name = diffusers_name.replace("proj.in", "proj_in")
            diffusers_name = diffusers_name.replace("proj.out", "proj_out")
            diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")

            # SDXL specificity.
            if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
                pattern = r"\.\d+(?=\D*$)"
                diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
            if ".in." in diffusers_name:
                diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
            if ".out." in diffusers_name:
                diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
            if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
                diffusers_name = diffusers_name.replace("op", "conv")
            if "skip" in diffusers_name:
                diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

            # LyCORIS specificity.
            if "time.emb.proj" in diffusers_name:
                diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
            if "conv.shortcut" in diffusers_name:
                diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")

            # General coverage.
            if "transformer_blocks" in diffusers_name:
                if "attn1" in diffusers_name or "attn2" in diffusers_name:
                    diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                    diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif "ff" in diffusers_name:
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            elif any(key in diffusers_name for key in ("proj_in", "proj_out")):
                unet_state_dict[diffusers_name] = state_dict.pop(key)
                unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            else:
                unet_state_dict[diffusers_name] = state_dict.pop(key)
                unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

        elif lora_name.startswith("lora_te_"):
            diffusers_name = key.replace("lora_te_", "").replace("_", ".")
            diffusers_name = diffusers_name.replace("text.model", "text_model")
            diffusers_name = diffusers_name.replace("self.attn", "self_attn")
            diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
            diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
            diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
            diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
            if "self_attn" in diffusers_name:
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            elif "mlp" in diffusers_name:
                # Be aware that this is the new diffusers convention and the rest of the code might
                # not utilize it yet.
                diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

        # (sayakpaul): Duplicate code. Needs to be cleaned.
        elif lora_name.startswith("lora_te1_"):
            diffusers_name = key.replace("lora_te1_", "").replace("_", ".")
            diffusers_name = diffusers_name.replace("text.model", "text_model")
            diffusers_name = diffusers_name.replace("self.attn", "self_attn")
            diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
            diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
            diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
            diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
            if "self_attn" in diffusers_name:
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            elif "mlp" in diffusers_name:
                # Be aware that this is the new diffusers convention and the rest of the code might
                # not utilize it yet.
                diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

        # (sayakpaul): Duplicate code. Needs to be cleaned.
        elif lora_name.startswith("lora_te2_"):
            diffusers_name = key.replace("lora_te2_", "").replace("_", ".")
            diffusers_name = diffusers_name.replace("text.model", "text_model")
            diffusers_name = diffusers_name.replace("self.attn", "self_attn")
            diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
            diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
            diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
            diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
            if "self_attn" in diffusers_name:
                te2_state_dict[diffusers_name] = state_dict.pop(key)
                te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            elif "mlp" in diffusers_name:
                # Be aware that this is the new diffusers convention and the rest of the code might
                # not utilize it yet.
                diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                te2_state_dict[diffusers_name] = state_dict.pop(key)
                te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

        # Rename the alphas so that they can be mapped appropriately.
        if lora_name_alpha in state_dict:
            alpha = state_dict.pop(lora_name_alpha).item()
            if lora_name_alpha.startswith("lora_unet_"):
                prefix = "unet."
            elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
                prefix = "text_encoder."
            else:
                prefix = "text_encoder_2."
            new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
            network_alphas.update({new_name: alpha})

    if len(state_dict) > 0:
        raise ValueError(f"The following keys have not been correctly be renamed: \n\n {', '.join(state_dict.keys())}")

    logger.info("Kohya-style checkpoint detected.")
    unet_state_dict = {f"{unet_name}.{module_name}": params for module_name, params in unet_state_dict.items()}
    te_state_dict = {f"{text_encoder_name}.{module_name}": params for module_name, params in te_state_dict.items()}
    te2_state_dict = (
        {f"text_encoder_2.{module_name}": params for module_name, params in te2_state_dict.items()}
        if len(te2_state_dict) > 0
        else None
    )
    if te2_state_dict is not None:
        te_state_dict.update(te2_state_dict)

    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict, network_alphas


def _convert_non_diffusers_qwen_lora_to_diffusers(state_dict):
    has_diffusion_model = any(k.startswith("diffusion_model.") for k in state_dict)
    if has_diffusion_model:
        state_dict = {k.removeprefix("diffusion_model."): v for k, v in state_dict.items()}

    has_lora_unet = any(k.startswith("lora_unet_") for k in state_dict)
    if has_lora_unet:
        state_dict = {k.removeprefix("lora_unet_"): v for k, v in state_dict.items()}

        def convert_key(key: str) -> str:
            prefix = "transformer_blocks"
            if "." in key:
                base, suffix = key.rsplit(".", 1)
            else:
                base, suffix = key, ""

            start = f"{prefix}_"
            rest = base[len(start) :]

            if "." in rest:
                head, tail = rest.split(".", 1)
                tail = "." + tail
            else:
                head, tail = rest, ""

            # Protected n-grams that must keep their internal underscores
            protected = {
                # pairs
                ("to", "q"),
                ("to", "k"),
                ("to", "v"),
                ("to", "out"),
                ("add", "q"),
                ("add", "k"),
                ("add", "v"),
                ("txt", "mlp"),
                ("img", "mlp"),
                ("txt", "mod"),
                ("img", "mod"),
                # triplets
                ("add", "q", "proj"),
                ("add", "k", "proj"),
                ("add", "v", "proj"),
                ("to", "add", "out"),
            }

            prot_by_len = {}
            for ng in protected:
                prot_by_len.setdefault(len(ng), set()).add(ng)

            parts = head.split("_")
            merged = []
            i = 0
            lengths_desc = sorted(prot_by_len.keys(), reverse=True)

            while i < len(parts):
                matched = False
                for L in lengths_desc:
                    if i + L <= len(parts) and tuple(parts[i : i + L]) in prot_by_len[L]:
                        merged.append("_".join(parts[i : i + L]))
                        i += L
                        matched = True
                        break
                if not matched:
                    merged.append(parts[i])
                    i += 1

            head_converted = ".".join(merged)
            converted_base = f"{prefix}.{head_converted}{tail}"
            return converted_base + (("." + suffix) if suffix else "")

        state_dict = {convert_key(k): v for k, v in state_dict.items()}

    has_default = any("default." in k for k in state_dict)
    if has_default:
        state_dict = {k.replace("default.", ""): v for k, v in state_dict.items()}

    converted_state_dict = {}
    all_keys = list(state_dict.keys())
    down_key = ".lora_down.weight"
    up_key = ".lora_up.weight"
    a_key = ".lora_A.weight"
    b_key = ".lora_B.weight"

    has_non_diffusers_lora_id = any(down_key in k or up_key in k for k in all_keys)
    has_diffusers_lora_id = any(a_key in k or b_key in k for k in all_keys)

    if has_non_diffusers_lora_id:

        def get_alpha_scales(down_weight, alpha_key):
            rank = down_weight.shape[0]
            alpha = state_dict.pop(alpha_key).item()
            scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
            scale_down = scale
            scale_up = 1.0
            while scale_down * 2 < scale_up:
                scale_down *= 2
                scale_up /= 2
            return scale_down, scale_up

        for k in all_keys:
            if k.endswith(down_key):
                diffusers_down_key = k.replace(down_key, ".lora_A.weight")
                diffusers_up_key = k.replace(down_key, up_key).replace(up_key, ".lora_B.weight")
                alpha_key = k.replace(down_key, ".alpha")

                down_weight = state_dict.pop(k)
                up_weight = state_dict.pop(k.replace(down_key, up_key))
                scale_down, scale_up = get_alpha_scales(down_weight, alpha_key)
                converted_state_dict[diffusers_down_key] = down_weight * scale_down
                converted_state_dict[diffusers_up_key] = up_weight * scale_up

    # Already in diffusers format (lora_A/lora_B), just pop
    elif has_diffusers_lora_id:
        for k in all_keys:
            if a_key in k or b_key in k:
                converted_state_dict[k] = state_dict.pop(k)
            elif ".alpha" in k:
                state_dict.pop(k)

    if len(state_dict) > 0:
        raise ValueError(f"`state_dict` should be empty at this point but has {state_dict.keys()=}")

    converted_state_dict = {f"transformer.{k}": v for k, v in converted_state_dict.items()}
    return converted_state_dict