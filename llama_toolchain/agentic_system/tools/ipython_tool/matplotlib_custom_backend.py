# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
A custom Matplotlib backend that overrides the show method to return image bytes.
"""

import base64
import io
import json as _json

import matplotlib
from matplotlib.backend_bases import FigureManagerBase

# Import necessary components from Matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg


class CustomFigureCanvas(FigureCanvasAgg):
    def show(self):
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        self.print_png(buf)
        image_bytes = buf.getvalue()
        buf.close()
        return image_bytes


class CustomFigureManager(FigureManagerBase):
    def __init__(self, canvas, num):
        super().__init__(canvas, num)


# Mimic module initialization that integrates with the Matplotlib backend system
def _create_figure_manager(num, *args, **kwargs):
    """
    Create a custom figure manager instance.
    """
    FigureClass = kwargs.pop("FigureClass", None)  # noqa: N806
    if FigureClass is None:
        from matplotlib.figure import Figure

        FigureClass = Figure  # noqa: N806
    fig = FigureClass(*args, **kwargs)
    canvas = CustomFigureCanvas(fig)
    manager = CustomFigureManager(canvas, num)
    return manager


def show():
    """
    Handle all figures and potentially return their images as bytes.

    This function iterates over all figures registered with the custom backend,
    renders them as images in bytes format, and could return a list of bytes objects,
    one for each figure, or handle them as needed.
    """
    image_data = []
    for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
        # Get the figure from the manager
        fig = manager.canvas.figure
        buf = io.BytesIO()  # Create a buffer for the figure
        fig.savefig(buf, format="png")  # Save the figure to the buffer in PNG format
        buf.seek(0)  # Go to the beginning of the buffer
        image_bytes = buf.getvalue()  # Retrieve bytes value
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data.append({"image_base64": image_base64})
        buf.close()

    req_con, resp_con = _open_connections()

    _json_dump = _json.dumps(
        {
            "type": "matplotlib",
            "image_data": image_data,
        }
    )
    req_con.send_bytes(_json_dump.encode("utf-8"))
    resp = _json.loads(resp_con.recv_bytes().decode("utf-8"))
    print(resp)


FigureCanvas = CustomFigureCanvas
FigureManager = CustomFigureManager
