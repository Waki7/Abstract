from grid_world.rendering.observation_rendering import *

CFG = {
    'global_resolution': [5, 5],
    'observation_resolution': [5, 5]
}


def test_draw():
    print()
    drawer = ObservationRenderer(CFG)
    drawer.reset_drawing()
    drawing = drawer.draw_circle(center=(2,2), radius=2)
    frame = drawer.get_frame((0, 0))
    print(drawing)
    print(frame)


