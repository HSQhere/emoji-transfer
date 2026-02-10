# coding: utf-8

import torch
import torch.utils.checkpoint
from typing import Any, Dict, Optional, Tuple, Union

from einops import rearrange

from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from .hm_adapters import CopyWeights, InsertReferenceAdapter

logger = logging.get_logger(__name__)

class HM3Denoising3D(UNet2DConditionModel, CopyWeights, InsertReferenceAdapter):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        reference_hidden_states: Optional[dict] = None,
        control_hidden_states: Optional[dict] = None,
        motion_pad_hidden_states: Optional[dict] = None,
        use_motion: bool = False,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:

        default_overall_up_factor = 2**self.num_upsamplers


        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0


        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        num_frames = sample.shape[2]
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)

        if not added_cond_kwargs is None:
            if 'image_embeds' in added_cond_kwargs:
                if isinstance(added_cond_kwargs['image_embeds'], torch.Tensor):
                    added_cond_kwargs['image_embeds'] = added_cond_kwargs['image_embeds'].repeat_interleave(repeats=num_frames, dim=0)
                else:
                    added_cond_kwargs['image_embeds'] = [x.repeat_interleave(repeats=num_frames, dim=0) for x in added_cond_kwargs['image_embeds']]

        if len(encoder_hidden_states.shape) == 3:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)
        elif len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, "b f l d -> (b f) l d")

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )


        sample = rearrange(sample, "b c f h w -> (b f) c h w")
        sample = self.conv_in(sample)


        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}


        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:

            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None

        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        res_cache = dict()
        down_block_res_samples = (sample,)
        for idx, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            res_cache[f"down_{idx}"] = sample.clone()
            if not control_hidden_states is None and f'down3_{idx}' in control_hidden_states:
                sample += rearrange(control_hidden_states[f'down3_{idx}'], "b c f h w -> (b f) c h w")
            if hasattr(self, 'motion_down') and use_motion:
                sample = self.motion_down[idx](sample,
                              None if motion_pad_hidden_states is None else motion_pad_hidden_states[f'down_{idx}'],
                              emb, num_frames)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples


        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)


            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            res_cache[f"up_{i}"] = sample.clone()
            if not control_hidden_states is None and f'up3_{i}' in control_hidden_states:
                sample += rearrange(control_hidden_states[f'up3_{i}'], "b c f h w -> (b f) c h w")
            if hasattr(self, "reference_modules_up") and not reference_hidden_states is None and f'up_{i}' in reference_hidden_states:
                sample = self.reference_modules_up[i](sample, reference_hidden_states[f'up_{i}'], num_frames)
            if hasattr(self, 'motion_up') and use_motion:
                sample = self.motion_up[i](sample,
                              None if motion_pad_hidden_states is None else motion_pad_hidden_states[f'up_{i}'],
                              emb, num_frames)

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:

            unscale_lora_layers(self, lora_scale)

        sample = rearrange(sample, "(b f) c h w -> b c f h w", f=num_frames)

        if not return_dict:
            return (sample, res_cache)

        return (UNet2DConditionOutput(sample=sample), res_cache)
