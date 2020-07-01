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
        'global_resolution': [200, 200],
        'observation_window': [399, 399],
        'observation_resolution': [399, 399]
    }
    print('---test_gif_rendering---')
    drawer = ObservationRenderer(cfg)
    # drawer.draw_diamond(center=(87.35, 34.9), apothem=10)

    circle_moves = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (-100, -100)]
    circle_point = (55.7, 65.35)
    diamond_moves = [(1, -1), (1, -1), (1, -1), (-1, 0), (-1, 0), (-1, 0), (-1, -1), (-100, -100)]
    diamond_point = (50., 56.)
    scale = 10
    agent_frames = []
    env_frames = []
    for circle_move, diamond_move in zip(circle_moves, diamond_moves):
        drawer.reset_drawing()
        drawer.draw_diamond(center=diamond_point, apothem=5.)
        drawer.draw_circle(center=circle_point, radius=10.)
        env_frame = drawer.get_drawing()
        env_frame = model_utils.convert_to_rgb_format(env_frame)
        env_frames.append(env_frame)
        print(circle_point)
        agent_frame = drawer.get_frame_at_point(center=circle_point)
        agent_frame = model_utils.convert_to_rgb_format(agent_frame)
        agent_frames.append(agent_frame)
        circle_point = (circle_point[0] + scale * circle_move[0], circle_point[1] + scale * circle_move[1])
        diamond_point = (diamond_point[0] + scale * diamond_move[0], diamond_point[1] + scale * diamond_move[1])

    write_gif(agent_frames, 'agent_render_test.gif', fps=2)
    # write_gif(env_frames, 'env_render_test.gif', fps=2)
