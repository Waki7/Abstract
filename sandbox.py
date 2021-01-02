import torch


def sim():
    precision = .000000001

    kill_odds = torch.Tensor([1 / 6, ]).to(torch.float64)
    cont_odds = torch.Tensor([5 / 6, ]).to(torch.float64)

    p1_odds = torch.Tensor([0.0, ]).to(torch.float64)
    p2_odds = torch.Tensor([0.0, ]).to(torch.float64)
    scenario_odds = torch.Tensor([1.0, ]).to(torch.float64)

    p1_odds += kill_odds * scenario_odds
    scenario_odds *= cont_odds

    p2_odds += kill_odds * scenario_odds
    scenario_odds *= cont_odds
    p2_odds += kill_odds * scenario_odds
    scenario_odds *= cont_odds

    i = 3
    while True:
        p1_odds += kill_odds * scenario_odds
        scenario_odds *= cont_odds

        p2_odds += kill_odds * scenario_odds
        scenario_odds *= cont_odds
        i += 2

        if scenario_odds < precision:
            print(
                'player 1 wins with odds {} and player 2 with odds {}, odds of no one dying by iteration {} is {}'.format(
                    p1_odds.item(), p2_odds.item(), i, scenario_odds.item()))
            print(scenario_odds)
            break


if __name__ == '__main__':
    sim()
