import logging
import gradio as gr
from modules import scripts
from RescaleCFG.nodes_RescaleCFG import RescaleCFG

class RescaleCFGScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.multiplier = 0.7

    sorting_priority = 15

    def title(self):
        return "RescaleCFG for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for RescaleCFG.</i></p>")
            enabled = gr.Checkbox(label="Enable RescaleCFG", value=self.enabled)
            multiplier = gr.Slider(label="RescaleCFG Multiplier", minimum=0.0, maximum=1.0, step=0.01, value=self.multiplier)

        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled]
        )

        return (enabled, multiplier)

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 2:
            self.enabled, self.multiplier = args[:2]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        if not self.enabled:
            # Reset the unet to its original state
            p.sd_model.forge_objects.unet = unet
            return

        unet = RescaleCFG().patch(unet, self.multiplier)[0]

        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params.update({
            "rescale_cfg_enabled": True,
            "rescale_cfg_multiplier": self.multiplier,
        })

        logging.debug(f"RescaleCFG: Enabled: {self.enabled}, Multiplier: {self.multiplier}")

        return