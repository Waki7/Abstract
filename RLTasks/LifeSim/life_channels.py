from enum import Enum
import numpy as np

channel_names = {
    0: "see",
    1: "hear",
    2: "speak",
    3: "feel",
    4: "movement"
}

speak_to_hear_map = {
    1: 1,  # say cry to hear cry
    2: 2,  # say food to hear food
    3: 3  # say water to hear water
}

move_to_see_map = {

}


class Speak(Enum):  # keep nothing as 0
    nothing = 0
    cry = 1
    food = 2
    water = 3
    # mom = 4
    # i = 5
    # want = 6
    # give = 7
    # me = 8


class Hear(Enum):  # keep nothing as 0
    nothing = 0
    cry = 1
    food = 2
    water = 3
    self = 4
    sibling = 5
    father = 6
    bird = 7
    mother = 8


class See(Enum):  # keep nothing as 0
    nothing = 0
    water = 1
    food = 2
    baby = 3
    smiles = 4
    cry = 5
    mom = 6
    room1 = 7
    room2 = 8
    outisde = 9
    sibling = 10
    come = 11
    leave = 12
    food_close = 13
    water_close = 14
    mom_close = 15
    sibling_close = 16


class Feel(Enum):  # keep nothing as 0
    # todo if modified then modify adjancent_reward_list
    nothing = 0
    drank = 1
    fed = 2
    thirsty = 3
    hungry = 4
    happy = 5
    content = 6
    discomfort = 7


class Movement(Enum):
    nothing = 0
    move_room1 = 1
    move_room2 = 2
    move_outside = 3
    wave = 3
    hit = 4
    grab = 5
    drink = 6
    feed = 7


class Channels(Enum):
    see = See
    hear = Hear
    speak = Speak
    feel = Feel
    movement = Movement


ChannelEnums = [See, Hear, Speak, Feel, Movement]

AGENT_STATE_CHANNELS = [See, Hear, Feel, Movement]
AGENT_ACTION_CHANNELS = [Speak, Movement]


def encode(msg, channel):
    import numpy as np
    encoded = np.zeros((1, len(channel)))
    if isinstance(msg, list):
        for i in msg:
            encoded[0, i.value] = 1.0
    else:
        encoded[0, msg.value] = 1.0
    return encoded


def getNeuronChannel(neuron):
    channel = 3
    if 'see' in neuron.name:
        channel = 0
    if 'hear' in neuron.name:
        channel = 1
    if 'speak' in neuron.name:
        channel = 2
    return channel


def encode_from_map(map, channels):
    encoded_input = []
    for channel in channels:
        channel_vals = map[channel]
        encoded_input.append(encode(channel_vals, channel))
    return np.concatenate(encoded_input, axis=1)

def decode_to_enum(vector, channels):
    index = 0
    max = np.argmax(vector)
    for c in channels:
        index += len(c)
        if max < index:
            return c(max)
