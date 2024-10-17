import logging
import gradio as gr
from modules import scripts
import k_diffusion
from advanced_model_sampling.nodes_model_advanced import (
    ModelSamplingDiscrete, ModelSamplingContinuousEDM, ModelSamplingContinuousV,
    ModelSamplingStableCascade, ModelSamplingSD3, ModelSamplingAuraFlow, ModelSamplingFlux
)

class AdvancedModelSamplingScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.sampling_mode = "Discrete"
        self.discrete_sampling = "eps"
        self.discrete_zsnr = False

    sorting_priority = 15

    def title(self):
        return "Advanced Model Sampling for Forge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Advanced Model Sampling.</i></p>")

            enabled = gr.Checkbox(label="Enable Advanced Model Sampling", value=self.enabled)

            sampling_mode = gr.Radio(
                ["Discrete"],
                label="Sampling Mode",
                value=self.sampling_mode
            )

            with gr.Group() as discrete_group:
                discrete_sampling = gr.Radio(
                    ["eps", "v_prediction", "lcm", "x0"],
                    label="Discrete Sampling Type",
                    value=self.discrete_sampling
                )
                discrete_zsnr = gr.Checkbox(label="Zero SNR", value=self.discrete_zsnr)

            def update_visibility(mode):
                return (
                    gr.Group.update(visible=(mode == "Discrete")),
                )

            sampling_mode.change(
                update_visibility,
                inputs=[sampling_mode],
                outputs=[discrete_group]
            )

        return (enabled, sampling_mode, discrete_sampling, discrete_zsnr)

def process_before_every_sampling(self, p, *args, **kwargs):
    if len(args) >= 4:  # Adjusted to reflect the relevant arguments count
        self.enabled, self.sampling_mode, self.discrete_sampling, self.discrete_zsnr = args[:4]
    else:
        logging.warning("Not enough arguments provided to process_before_every_sampling")
        return

    if not self.enabled:
        return

    # Clone the UNet model
    unet = p.sd_model.forge_objects.unet.clone()

    # Apply the discrete sampling mode
    if self.sampling_mode == "Discrete":
        unet = ModelSamplingDiscrete().patch(unet, self.discrete_sampling, self.discrete_zsnr)[0]

    # Update the UNet model with the modified version
    p.sd_model.forge_objects.unet = unet

    # Update generation parameters for logging/debugging
    p.extra_generation_params.update({
        "advanced_sampling_enabled": self.enabled,
        "advanced_sampling_mode": self.sampling_mode,
        "discrete_sampling": self.discrete_sampling if self.sampling_mode == "Discrete" else None,
        "discrete_zsnr": self.discrete_zsnr if self.sampling_mode == "Discrete" else None,
    })

    # Log the current sampling configuration
    logging.debug(f"Advanced Model Sampling: Enabled: {self.enabled}, Mode: {self.sampling_mode}")

    return

