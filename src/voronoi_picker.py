# -*- coding: utf-8 -*-
# vispy: gallery 30
# Modified from vispy demo application.
"""
Computing a Voronoi diagram on the GPU. Shows how to use uniform arrays.
Original version by Xavier Olive (xoolive).
"""

import numpy as np

from vispy import app
from vispy import gloo

# Voronoi shaders.
VS_voronoi = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0., 1.);
}
"""

FS_voronoi = """
uniform int num;
uniform vec2 u_seeds[500];
uniform vec3 u_colors[500];
uniform vec2 u_screen;
void main() {
    float dist = distance(u_screen * u_seeds[0], gl_FragCoord.xy);
    vec3 color = u_colors[0];
    for (int i = 1; i < num; i++) {
        float current = distance(u_screen * u_seeds[i], gl_FragCoord.xy);
        if (current < dist) {
            color = u_colors[i];
            dist = current;
        }
    }
    gl_FragColor = vec4(color, 1.0);
}
"""

FS_voronoi_backup = """
uniform vec2 u_seeds[32];
uniform vec3 u_colors[32];
uniform vec2 u_screen;
void main() {
    float dist = distance(u_screen * u_seeds[0], gl_FragCoord.xy);
    vec3 color = u_colors[0];
    for (int i = 1; i < 32; i++) {
        float current = distance(u_screen * u_seeds[i], gl_FragCoord.xy);
        if (current < dist) {
            color = u_colors[i];
            dist = current;
        }
    }
    gl_FragColor = vec4(color, 1.0);
}
"""

# Seed point shaders.
VS_seeds = """
attribute vec2 a_position;
uniform float u_ps;
void main() {
    gl_Position = vec4(2. * a_position - 1., 0., 1.);
    gl_PointSize = 10. * u_ps;
}
"""

FS_seeds = """
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(0., 0., 0., 1.);
}
"""


class Canvas(app.Canvas):
    def __init__(self, size=(600, 600), title="Voronoi Picker", 
                 input_points=None, buffer_points=None):
        app.Canvas.__init__(self, size=size, title=title, keys="interactive")

        self.width = 0
        self.height = 0
        self.ps = self.pixel_scale

        if input_points is None:
            # App demo. Random display of 31 points
            self.max_idx = 32
            self.min_idx = 31
            self.seeds = np.random.uniform(0, 1.0 * self.ps,
                                           size=(self.max_idx - 1, 2))
        else:
            self.min_idx = len(input_points)
            if buffer_points is None:
                # No buffer points provided. Start with only real points
                self.max_idx = self.min_idx + 1
                self.seeds = input_points
            else:
                # Buffer points provided. Program should confirm or fine-tune 
                # buffer placement.
                self.max_idx = self.min_idx + len(buffer_points) + 1
                self.seeds = np.append(input_points, buffer_points, axis=0)

        self.idx = self.max_idx - 1

        # Current max # of points is 500. Pre-declare for gloo program, and 
        # hide off-screen at (-1, -1)
        # It WAS 800, but for some reason that broke. Now it's 500 and it 
        # hopefully works forever
        self.seeds = np.append(
            self.seeds, 
            np.ones([501 - self.max_idx, 2]) * -1, axis=0).astype(np.float32)
        # Assign colors to cells. Real cells colored a shade of red. 
        # Buffer cells colored white
        self.colors = np.zeros((500, 3)).astype(np.float32)
        self.colors[0:self.min_idx, 0] = 1 * np.linspace(0.5, 1, self.min_idx)
        self.colors[self.min_idx:, :] = (1, 1, 1)

        # Set Voronoi program.
        self.program_v = gloo.Program(VS_voronoi, FS_voronoi)
        self.program_v["num"] = self.max_idx
        self.program_v["a_position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        # HACK: work-around a bug related to uniform arrays until
        # issue #345 is solved.
        for i in range(500):
            self.program_v["u_seeds[%d]" % i] = self.seeds[i, :]
            self.program_v["u_colors[%d]" % i] = self.colors[i, :]

        # Set seed points program.
        self.program_s = gloo.Program(VS_seeds, FS_seeds)
        self.program_s["a_position"] = self.seeds
        self.program_s["u_ps"] = self.ps

        self.activate_zoom()

        self.show()

    def on_draw(self, _event):
        gloo.clear()
        self.program_v.draw("triangle_strip")
        self.program_s.draw("points")

    def on_resize(self, _event):
        self.activate_zoom()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program_v["u_screen"] = self.physical_size

    def on_mouse_move(self, event):
        x, y = event.pos
        x, y = x / float(self.width), 1 - y / float(self.height)

        self.program_v["u_seeds[%d]" % self.idx] = x * self.ps, y * self.ps
        self.seeds[self.idx, :] = x, y

        self.program_s["a_position"].set_data(self.seeds)
        self.update()

    def on_mouse_press(self, event):
        if event.button == 1:
            # Add a point
            self.max_idx = self.max_idx + 1
            self.idx = self.max_idx - 1
            self.program_v["num"] = self.max_idx
        elif event.button == 2:
            # Prevent overwriting real points
            if self.idx > self.min_idx:
                # Remove point
                self.seeds[self.idx, :] = [-1, -1]
                self.max_idx = self.max_idx - 1
                self.idx = self.max_idx - 1
                self.program_v["num"] = self.max_idx


def pick_points(size=(600, 600), input_points=None, buffer_points=None):
    c = Canvas(size=size, input_points=input_points, 
               buffer_points=buffer_points)
    app.run()
    app.quit()
    buffer_points = c.seeds[c.min_idx:c.idx, :]
    del c
    return buffer_points


if __name__ == "__main__":
    c = Canvas()
    app.run()
    app.quit()
    print(c.seeds[:c.idx, :])
    print("{} points".format(len(c.seeds[:c.idx, :])))
