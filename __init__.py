"""JR Drama 3 local nodes for ComfyUI. Built with Fish Audio."""
WEB_DIRECTORY = "./web"


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension
    from . import jr_analysis, jr_delivery, jr_generate, jr_script, jr_live, jr_benchmark, jr_edit_region, jr_realtime

    class DramaExtension(ComfyExtension):
        async def get_node_list(self):
            return [*jr_generate.NODES, *jr_script.NODES, *jr_delivery.NODES, *jr_analysis.NODES,
                    jr_live.LiveText, jr_realtime.RealtimeSpeech, jr_edit_region.EditRegion, *jr_benchmark.NODES]

        async def on_load(self):
            jr_live.register_canvas_control()

    return DramaExtension()
