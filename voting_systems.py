from graph import Graph
import copy
import numpy as np


def rank_ranked_pairs(comparisons):
    ranked = []
    groups = [] # used to identify ties

    group = 0
    has_inconsistency = False
    while True:
        winners, has_inconsistency_ = single_ranked_pairs(copy.deepcopy(comparisons))
        has_inconsistency = has_inconsistency_ or has_inconsistency
        if len(winners) > 1:
            winners_sum = []
            for winner in winners:
                sum = 0
                for id_ in comparisons[winner]:
                    sum += comparisons[winner][id_]
                winners_sum.append(sum)
            winners_sum, winners = zip(*sorted(zip(winners_sum, winners)))
            winners = list(winners)[::-1]

        for id in winners:
            del comparisons[id]
            for id_ in comparisons:
                del comparisons[id_][id]

        # ranked += [winners]
        for id in winners:
            ranked.append(id)
            groups.append(group)
        group -= 1

        if comparisons == {}:
            break

    return ranked, groups, has_inconsistency


def single_ranked_pairs(comparisons):
    graph = Graph(comparisons.keys())
    has_inconsistency = False
    while True:
        max_maj = float('-inf')
        max_min = float('-inf')
        can_a = ""
        can_b = ""

        for can_i in comparisons:
            for can_j in comparisons:
                if can_i != can_j and (comparisons[can_i][can_j] > comparisons[can_j][can_i]) and \
                (comparisons[can_i][can_j] > max_maj or \
                (comparisons[can_i][can_j] == max_maj and comparisons[can_j][can_i] > max_min)):
                    max_maj = comparisons[can_i][can_j]
                    max_min = comparisons[can_j][can_i]
                    can_a = can_i
                    can_b = can_j

        if max_maj == float('-inf') and max_min == float('-inf'):
            break

        graph.add_edge(can_a, can_b)
        if (graph.has_cycle(can_a) or graph.has_cycle(can_b)):
            has_inconsistency = True
            graph.remove_edge(can_a, can_b)

        comparisons[can_a][can_b] = -1
        comparisons[can_b][can_a] = -1

    return graph.get_sources(), has_inconsistency


def rank_schulze(comparisons):
    strengths = copy.deepcopy(comparisons)

    for can_i in comparisons.keys():
        for can_j in comparisons.keys():
            if can_i != can_j:
                for can_k in comparisons.keys():
                    if can_i != can_k and can_j != can_k:
                        strengths[can_j][can_k] = max(strengths[can_j][can_k], min(strengths[can_j][can_i], strengths[can_i][can_k]))

    number_of_victories = {can: 0 for can in comparisons}
    for can_i in strengths:
        for can_j in strengths:
            if can_i != can_j and strengths[can_i][can_j] > strengths[can_j][can_i]:
                number_of_victories[can_i] += 1

    return tuple(zip(*sorted(zip(number_of_victories.values(), number_of_victories.keys()), reverse=True)))[::-1]


def rank_black(comparisons):
    ranked = []
    groups = [] # used to identify ties

    group = 0
    while True:
        winners = single_black(copy.deepcopy(comparisons))
        if len(winners) > 1:
            winners_sum = []
            for winner in winners:
                sum = 0
                for id_ in comparisons[winner]:
                    sum += comparisons[winner][id_]
                winners_sum.append(sum)
            winners_sum, winners = zip(*sorted(zip(winners_sum, winners)))
            winners = list(winners)[::-1]

        for id in winners:
            del comparisons[id]
            for id_ in comparisons:
                del comparisons[id_][id]

        for id in winners:
            ranked.append(id)
            groups.append(group)
        group -= 1

        if comparisons == {}:
            break

    return ranked, groups

def single_black(comparisons):
    borda_count = {}
    n_candidates = len(comparisons.keys())
    is_condorcet = {can: True for can in comparisons}
    for can_i in comparisons:
        sum = 0
        for can_j in comparisons:
            if can_i == can_j: continue
            sum += comparisons[can_i][can_j]
            if comparisons[can_j][can_i] >= comparisons[can_i][can_j]:
                is_condorcet[can_i] = False
        borda_count[can_i] = sum

    for can in is_condorcet:
        if is_condorcet[can]:
            return [can]

    max_borda_count = max(borda_count.values())
    return [can for can in borda_count if borda_count[can] == max_borda_count]


def rank_ctas(scores, comparisons):
    ranked = []
    groups = [] # used to identify ties

    group = 0
    while True:
        winners = single_ctas(copy.deepcopy(scores), copy.deepcopy(comparisons))
        if len(winners) > 1:
            winners_sum = []
            for winner in winners:
                sum = 0
                for id_ in comparisons[winner]:
                    sum += comparisons[winner][id_]
                winners_sum.append(sum)
            winners_sum, winners = zip(*sorted(zip(winners_sum, winners)))
            winners = list(winners)[::-1]

        for id in winners:
            del comparisons[id]
            del scores[id]
            for id_ in comparisons:
                del comparisons[id_][id]

        for id in winners:
            ranked.append(id)
            groups.append(group)
        group -= 1

        if comparisons == {}:
            break

    return ranked, groups


def single_ctas(scores, comparisons):
    "winner is condorcet if they exist, otherwise use highest score"
    n_candidates = len(comparisons.keys())
    is_condorcet = {can: True for can in comparisons}
    for can_i in comparisons:
        sum = 0
        for can_j in comparisons:
            if can_i == can_j: continue
            sum += comparisons[can_i][can_j]
            if comparisons[can_j][can_i] >= comparisons[can_i][can_j]:
                is_condorcet[can_i] = False

    for can in is_condorcet:
        if is_condorcet[can]:
            return [can]

    max_score = max(scores.values())
    return [can for can in scores if scores[can] == max_score]


def rank_stas(scores, comparisons):
    ranked = []
    groups = [] # used to identify ties

    group = 0
    while True:
        winners = single_stas(copy.deepcopy(scores), copy.deepcopy(comparisons))
        if len(winners) > 1:
            winners_sum = []
            for winner in winners:
                sum = 0
                for id_ in comparisons[winner]:
                    sum += comparisons[winner][id_]
                winners_sum.append(sum)
            winners_sum, winners = zip(*sorted(zip(winners_sum, winners)))
            winners = list(winners)[::-1]

        for id in winners:
            del comparisons[id]
            del scores[id]
            for id_ in comparisons:
                del comparisons[id_][id]

        for id in winners:
            ranked.append(id)
            groups.append(group)
        group -= 1

        if comparisons == {}:
            break

    return ranked, groups


def single_stas(scores, comparisons):
    "winner is the candidate with the highest score from the smith set"
    cannot_beat = {can_i: {can_j: comparisons[can_i][can_j] <= comparisons[can_j][can_i] for can_j in comparisons} for can_i in comparisons}
    for can_k in comparisons:
        for can_i in comparisons:
            for can_j in comparisons:
                cannot_beat[can_i][can_j] = cannot_beat[can_i][can_j] or (cannot_beat[can_i][can_k] and cannot_beat[can_k][can_j])

    possible_smith_sets = []

    for can_i in comparisons:
        possible_smith_set = {can_i}
        for can_j in comparisons:
            if cannot_beat[can_i][can_j]:
                possible_smith_set.add(can_j)
        possible_smith_sets.append(possible_smith_set)

    smith_set = min(possible_smith_sets, key=len)

    max_score = max([scores[can] for can in smith_set])
    return [can for can in smith_set if scores[can] == max_score]


def rank_condorcet_hare(ballots, comparisons, total_votes):
    ranked = []
    groups = [] # used to identify ties

    group = 0
    while True:
        winners = single_condorcet_hare(copy.deepcopy(ballots), copy.deepcopy(comparisons), total_votes)
        if len(winners) > 1:
            winners_sum = []
            for winner in winners:
                sum = 0
                for id_ in comparisons[winner]:
                    sum += comparisons[winner][id_]
                winners_sum.append(sum)
            winners_sum, winners = zip(*sorted(zip(winners_sum, winners)))
            winners = list(winners)[::-1]

        for id in winners:
            del comparisons[id]
            for id_ in comparisons:
                del comparisons[id_][id]
            new_ballots = {}
            for ballot in ballots:
                new_ballot = ballot.replace(id + "-", "")
                if new_ballot in new_ballots.keys():
                    new_ballots[new_ballot] += ballots[ballot]
                else:
                    new_ballots[new_ballot] = ballots[ballot]
            ballots = new_ballots

        for id in winners:
            ranked.append(id)
            groups.append(group)
        group -= 1

        if comparisons == {}:
            break

    return ranked, groups


def single_condorcet_hare(ballots, comparisons, total_votes):
    """
    iteratively applies the Hare method (IRV) to the Smith Set
    """

    while True:
        cannot_beat = {can_i: {can_j: comparisons[can_i][can_j] <= comparisons[can_j][can_i] for can_j in comparisons} for can_i in comparisons}
        for can_k in comparisons:
            for can_i in comparisons:
                for can_j in comparisons:
                    cannot_beat[can_i][can_j] = cannot_beat[can_i][can_j] or (cannot_beat[can_i][can_k] and cannot_beat[can_k][can_j])

        possible_smith_sets = []

        for can_i in comparisons:
            possible_smith_set = {can_i}
            for can_j in comparisons:
                if cannot_beat[can_i][can_j]:
                    possible_smith_set.add(can_j)
            possible_smith_sets.append(possible_smith_set)

        smith_set = min(possible_smith_sets, key=len)

        if len(smith_set) == 1:
            return [smith_set.pop()]

        smith_ballots = {}
        for ballot in ballots:
            new_ballot = ballot
            for can in comparisons:
                if can not in smith_set:
                    new_ballot = new_ballot.replace(can + "-", "")
            if new_ballot in smith_ballots.keys():
                smith_ballots[new_ballot] += ballots[ballot]
            else:
                smith_ballots[new_ballot] = ballots[ballot]

        votes = {}
        for ballot in smith_ballots:
            first_choice = ballot.split('-')[0]
            if first_choice in votes.keys():
                votes[first_choice] += smith_ballots[ballot]
            else:
                votes[first_choice] = smith_ballots[ballot]

        least_votes = np.min(list(votes.values()))
        remaining_candidates = []
        for candidate in list(votes.keys()):
            if votes[candidate] != least_votes:
                remaining_candidates.append(candidate)

        if len(remaining_candidates) == 0:
            return list(smith_set)
        if len(remaining_candidates) == 1:
            return remaining_candidates

        ballots = {}
        for ballot in smith_ballots:
            new_ballot = ballot
            for can in comparisons:
                if can not in remaining_candidates:
                    new_ballot = new_ballot.replace(can + "-", "")
            if new_ballot in ballots.keys():
                ballots[new_ballot] += smith_ballots[ballot]
            else:
                ballots[new_ballot] = smith_ballots[ballot]


def rank_borda(comparisons, total_votes):
    borda_count = {}
    n_candidates = len(comparisons.keys())
    for can_i in comparisons:
        sum = 0
        for can_j in comparisons:
            sum += comparisons[can_i][can_j]
        if total_votes > 0: # vestigial code
            weighted_sum = sum / (total_votes * (n_candidates * (n_candidates - 1)) / 2)
        else:
            weighted_sum = 0
        borda_count[can_i] = sum

    return tuple(zip(*sorted(zip(borda_count.values(), borda_count.keys()), reverse=True)))[::-1]


def rank_plurality(plurality, options, total_votes):
    sums = []
    ids = []
    sum_total = 0
    n_candidates = len(plurality.keys())
    for id in options:
        weighted_sum = 0
        if total_votes > 0 and id in plurality.keys():
            weighted_sum = plurality[id] / total_votes
        sums.append(weighted_sum)
        ids.append(id)

    sums, ids = zip(*sorted(zip(sums, ids)))
    return list(ids[::-1]), list(sums[::-1])


def rank_approval(approvals, total_votes):
    sums = []
    ids = []
    sum_total = 0
    n_candidates = len(approvals.keys())
    for id in approvals:
        weighted_sum = 0
        if total_votes > 0:
            weighted_sum = approvals[id] / total_votes
        sums.append(weighted_sum)
        ids.append(id)

    sums, ids = zip(*sorted(zip(sums, ids)))

    return list(ids[::-1]), list(sums[::-1])


def rank_instant_runoff(ballots, total_votes):
    ballots_ = []
    for ballot in ballots:
        ballots_.append((ballot.split('-')[:-1], ballots[ballot]))

    n_candidates = len(ballots_[0][0])
    ranked = []
    round_votes = []

    ballots_ = []
    for ballot in ballots:
        ballot_ = (ballot.split('-')[:-1], ballots[ballot])
        for candidate in ranked:
            ballot_[0].remove(candidate)
        ballots_.append(ballot_)

    while len(ballots_[0][0]) > 0:

        votes = {}
        for ballot_ in ballots_:
            if ballot_[0][0] in votes.keys():
                votes[ballot_[0][0]] += ballot_[1]
            else:
                votes[ballot_[0][0]] = ballot_[1]

        round_votes.append(votes)

        least_candidate = list(votes.keys())[np.argmin(list(votes.values()))]
        remove = [least_candidate]
        ranked.append(least_candidate)

        for i_ballot_, ballot_ in enumerate(ballots_):
            new_ballot_ = []
            for i in range(len(ballot_[0])):
                if ballot_[0][i] not in remove:
                    new_ballot_.append(ballot_[0][i])
            ballots_[i_ballot_] = (new_ballot_, ballot_[1])

    return ranked[::-1], round_votes
