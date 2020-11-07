from graph import Graph
import copy

def rank_ranked_pairs(comparisons):
    """
    returns a ranked ordering of the ranked pairs method
    """
    ranked = []
    groups = [] # used to identify ties

    group = 0
    while True:
        winners = VotingMethods.single_ranked_pairs(copy.deepcopy(comparisons))
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

    return ranked, groups

def single_ranked_pairs(comparisons):
    """
    returns the winner(s) of the ranked pairs method
    """
    graph = Graph(comparisons.keys())
    while True:
        max_maj = 0
        can_a = ""
        can_b = ""

        for can_i in comparisons:
            for can_j in comparisons:
                if can_i != can_j and comparisons[can_i][can_j] > comparisons[can_j][can_i]:
                    max_maj = comparisons[can_i][can_j]
                    can_a = can_i
                    can_b = can_j

        if max_maj == 0:
            break

        graph.add_edge(can_a, can_b)
        if (graph.has_cycle(can_a) or graph.has_cycle(can_b)):
            graph.remove_edge(can_a, can_b)

        comparisons[can_a][can_b] = -1
        comparisons[can_b][can_a] = -1

    return graph.get_sources()

def rank_borda(comparisons, total_votes):
    """
    returns a ranked ordering of the borda count method
    """
    sums = []
    ids = []
    sum_total = 0
    n_candidates = len(comparisons.keys())
    for id in comparisons:
        sum = 0
        for id_ in comparisons:
            sum += comparisons[id][id_]
        sum_total += sum
        if total_votes > 0:
            weighted_sum = sum / (total_votes * (n_candidates * (n_candidates - 1)) / 2)
        else:
            weighted_sum = 0
        sums.append(weighted_sum)
        ids.append(id)

    sums, ids = zip(*sorted(zip(sums, ids)))

    return list(ids[::-1]), list(sums[::-1])

def rank_plurality(plurality, options, total_votes):
    """
    returns a ranked ordering of the plurality method
    """
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
    """
    returns a ranked ordering of the approval method
    """
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
    """
    returns a ranked ordering of the approval method
    """
    ballots_ = []
    for ballot in ballots:
        ballots_.append((ballot.split('-')[:-1], ballots[ballot]))

    ranked = []
    groups = [] # used to identify ties
    group = 0
    votes = {}
    while len(ballots_[0][0]) > 0:
        for ballot_ in ballots_:
            if ballot_[0][0] in votes.keys():
                votes[ballot_[0][0]] += ballot_[1]
            else:
                votes[ballot_[0][0]] = ballot_[1]

        curRank = []
        for option in votes:
            if votes[option] > (total_votes / 2):
                curRank.append(option)
                break
            elif votes[option] == (total_votes / 2):
                curRank.append(option)

        remove = []
        if len(curRank) > 0:
            for option in curRank:
                ranked.append(option)
                groups.append(group)

                remove.append(option)

        else:
            least_options = []
            least_amount = 0
            for option in votes:
                if votes[option] < least_amount:
                    least_amount = votes[option]
                    least_options = [option]
                elif votes[option] == least_amount:
                    least_options.append(option)

            remove = least_options

            if len(remove) == len(votes):
                for option in curRank:
                    ranked.append(option)
                    groups.append(group)

        for i_ballot_, ballot_ in enumerate(ballots_):
            new_ballot_ = []
            for i in range(len(ballot_[0])):
                if ballot_[0][i] not in remove:
                    new_ballot_.append(ballot_[0][i])
            ballots_[i_ballot_] = (new_ballot_, ballot_[1])

        group -= 1
        votes = {}

    return ranked, groups
