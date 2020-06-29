from array2gif import write_gif

from grid_world.rendering.observation_rendering import *

CFG = {
    'global_resolution': [5, 5],
    'observation_resolution': [5, 5]
}


# def test_draw():
#     print('---test_draw---')
#     drawer = ObservationRenderer(CFG)
#     drawer.reset_drawing()
#     drawing = drawer.draw_diamond(center=(2, 2), length=2)
#     # drawing = drawer.draw_circle(center=(2,2), radius=2)
#     frame = drawer.get_frame((0, 0))
#     print('drawing:\n {}  \n'.format(drawing))
#     print('frame:\n {}  \n'.format(frame))


def test_gif_rendering():
    cfg = {
        'global_resolution': [100, 100],
        'observation_window': [199, 199],
        'observation_resolution': [99, 99]
    }
    print('---test_gif_rendering---')
    drawer = ObservationRenderer(cfg)
    drawer.reset_drawing()
    # drawer.draw_diamond(center=(87.35, 34.9), apothem=10)
    drawer.draw_diamond(center=(50., 56.), apothem=2.)
    new_frame1 = drawer.get_drawing()
    new_frame1 = model_utils.convert_to_rgb_format(new_frame1)

    drawer.reset_drawing()
    drawer.draw_diamond(center=(50.7, 56.72), apothem=2.)
    # drawer.draw_circle(center=(10.7, 56.35), radius=50)
    new_frame2 = drawer.get_drawing()
    new_frame2 = model_utils.convert_to_rgb_format(new_frame2)

    frames = [new_frame1, new_frame2]
    write_gif(frames, 'rgbbgr.gif', fps=2)
