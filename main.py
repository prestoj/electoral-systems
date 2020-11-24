import voting_systems as vs
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde
import pickle


np.random.seed(1)

CANDIDATES = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Heidi']
N_POPULATION = 1000
COLORS = px.colors.qualitative.Vivid[:len(CANDIDATES)]
LEFT_COLORS = ["rgb(4, 79, 199)", "rgb(72, 182, 250)", "rgb(40, 191, 201)", "rgb(86, 252, 222)"]
RIGHT_COLORS = ["rgb(189, 9, 9)", "rgb(237, 97, 50)", "rgb(219, 141, 24)", "rgb(252, 210, 71)"]


def update_rankings(df_population, df_candidates):
    population_dims = np.array([df_population['dim_1'], df_population['dim_2']]).T.reshape(N_POPULATION, 1, 2)
    candidate_dims = np.array([df_candidates['dim_1'], df_candidates['dim_2']]).T.reshape(1, len(CANDIDATES), 2)
    euclidean_distances = np.sqrt(np.sum(np.power((population_dims - candidate_dims), 2), axis=2))
    rankings = np.argsort(euclidean_distances)

    for i_pop in range(N_POPULATION):
        for i_candidate, candidate in enumerate(CANDIDATES):
            df_population['choice_' + str(i_candidate + 1)].iloc[i_pop] = CANDIDATES[rankings[i_pop, i_candidate]]
            df_population['distance_' + str(i_candidate + 1)].iloc[i_pop] = euclidean_distances[i_pop, rankings[i_pop, i_candidate]]

    for i_candidate in range(len(CANDIDATES)):
        df_candidates['mean_distance'].iloc[i_candidate] = euclidean_distances[:, i_candidate].mean()
        for i_choice in range(len(CANDIDATES)):
            df_candidates['choice_' + str(i_choice + 1)].iloc[i_candidate] = np.sum(rankings[:, i_choice] == i_candidate)

    return df_population, df_candidates


def update_rankings_partisan(df_population, df_candidates):
    left_population = df_population[df_population['dim_1'] < 0]
    right_population = df_population[df_population['dim_1'] > 0]
    left_candidates = df_candidates[df_candidates['dim_1'] < 0]
    right_candidates = df_candidates[df_candidates['dim_1'] > 0]

    left_population_dims = np.array([left_population['dim_1'], left_population['dim_2']]).T.reshape(-1, 1, 2)
    right_population_dims = np.array([right_population['dim_1'], right_population['dim_2']]).T.reshape(-1, 1, 2)
    left_candidate_dims = np.array([left_candidates['dim_1'], left_candidates['dim_2']]).T.reshape(1, len(CANDIDATES) // 2, 2)
    right_candidate_dims = np.array([right_candidates['dim_1'], right_candidates['dim_2']]).T.reshape(1, len(CANDIDATES) // 2, 2)
    left_euclidean_distances = np.sqrt(np.sum(np.power((left_population_dims - left_candidate_dims), 2), axis=2))
    right_euclidean_distances = np.sqrt(np.sum(np.power((right_population_dims - right_candidate_dims), 2), axis=2))
    left_rankings = np.argsort(left_euclidean_distances)
    right_rankings = np.argsort(right_euclidean_distances)

    for i_pop in range(len(left_population)):
        for i_candidate, candidate in enumerate(CANDIDATES[:len(CANDIDATES) // 2]):
            left_population['choice_' + str(i_candidate + 1)].iloc[i_pop] = CANDIDATES[left_rankings[i_pop, i_candidate]]
            left_population['distance_' + str(i_candidate + 1)].iloc[i_pop] = left_euclidean_distances[i_pop, left_rankings[i_pop, i_candidate]]

    for i_pop in range(len(right_population)):
        for i_candidate, candidate in enumerate(CANDIDATES[len(CANDIDATES) // 2:]):
            right_population['choice_' + str(i_candidate + 1)].iloc[i_pop] = CANDIDATES[len(CANDIDATES) // 2:][right_rankings[i_pop, i_candidate]]
            right_population['distance_' + str(i_candidate + 1)].iloc[i_pop] = right_euclidean_distances[i_pop, right_rankings[i_pop, i_candidate]]

    for i_candidate in range(len(CANDIDATES) // 2):
        left_candidates['mean_distance'].iloc[i_candidate] = left_euclidean_distances[:, i_candidate].mean()
        for i_choice in range(len(CANDIDATES) // 2):
            left_candidates['choice_' + str(i_choice + 1)].iloc[i_candidate] = np.sum(left_rankings[:, i_choice] == i_candidate)

    for i_candidate in range(len(CANDIDATES) // 2):
        right_candidates['mean_distance'].iloc[i_candidate] = right_euclidean_distances[:, i_candidate].mean()
        for i_choice in range(len(CANDIDATES) // 2):
            right_candidates['choice_' + str(i_choice + 1)].iloc[i_candidate] = np.sum(right_rankings[:, i_choice] == i_candidate)

    return pd.concat([left_population, right_population]), pd.concat([left_candidates, right_candidates])


def initialize_dfs(
    candidate_dim_1_values,
    candidate_dim_2_values,
    population_dim_1_values,
    population_dim_2_values
):
    df_population  = pd.DataFrame({
        'dim_1': population_dim_1_values,
        'dim_2': population_dim_2_values,
    })

    df_candidates  = pd.DataFrame({
        'dim_1': candidate_dim_1_values,
        'dim_2': candidate_dim_2_values,
        'mean_distance': -1,
    }, index=CANDIDATES)

    for i_candidate, candidate in enumerate(CANDIDATES):
        df_population['choice_' + str(i_candidate + 1)] = 0
        df_population['distance_' + str(i_candidate + 1)] = 0
        df_candidates['choice_' + str(i_candidate + 1)] = 0

    return update_rankings(df_population, df_candidates)


def initialize_dfs_partisan(
    candidate_dim_1_values,
    candidate_dim_2_values,
    population_dim_1_values,
    population_dim_2_values
):
    df_population  = pd.DataFrame({
        'dim_1': population_dim_1_values,
        'dim_2': population_dim_2_values,
    })

    df_candidates  = pd.DataFrame({
        'dim_1': candidate_dim_1_values,
        'dim_2': candidate_dim_2_values,
        'mean_distance': -1,
    }, index=CANDIDATES)

    for i_candidate, candidate in enumerate(CANDIDATES):
        df_population['choice_' + str(i_candidate + 1)] = 0
        df_population['distance_' + str(i_candidate + 1)] = 0
        df_candidates['choice_' + str(i_candidate + 1)] = 0

    return update_rankings_partisan(df_population, df_candidates)


def get_winners(df_population, df_candidates, electoral_system):
    if electoral_system == "plurality":
        return get_winners_plurality(df_candidates)
    elif electoral_system == "instant_runoff":
        return get_winners_instant_runoff(df_population, df_candidates)[0]
    elif electoral_system == "ranked_pairs":
        return get_winners_ranked_pairs(df_population, df_candidates)
    elif electoral_system == "borda":
        return get_winners_borda(df_population, df_candidates)


def get_winners_plurality(df_candidates):
    return df_candidates.index[df_candidates['choice_1'].argsort().to_list()].to_list()[::-1]


def get_winners_instant_runoff(df_population, df_candidates):
    ballots = {}
    for i_pop in range(len(df_population)):
        ballot = ""
        for i_choice in range(len(df_candidates)):
            ballot += df_population['choice_' + str(i_choice + 1)].iloc[i_pop]
            ballot += '-'

        if ballot in ballots.keys():
            ballots[ballot] += 1
        else:
            ballots[ballot] = 1

    return vs.rank_instant_runoff(ballots, len(df_population))


def get_winners_ranked_pairs(df_population, df_candidates, comparisons=None):
    if comparisons is None:
        comparisons = get_comparisons(df_population, df_candidates)

    return vs.rank_ranked_pairs(comparisons)[0]


def get_winners_borda(df_population, df_candidates):
    comparisons = {}
    for candidate_i in CANDIDATES:
        comparisons[candidate_i] = {}
        for candidate_j in CANDIDATES:
            comparisons[candidate_i][candidate_j] = 0

    for i_pop in range(N_POPULATION):
        ballot_candidates = df_population.iloc[i_pop][['choice_' + str(i_choice + 1) for i_choice in range(len(CANDIDATES))]].to_list()
        for i, candidate_i in enumerate(ballot_candidates[:-1]):
            for j, candidate_j in enumerate(ballot_candidates[i + 1:]):
                comparisons[candidate_i][candidate_j] += 1

    return vs.rank_borda(comparisons, N_POPULATION)[0]


def get_comparisons(df_population, df_candidates):
    comparisons = {}
    for candidate_i in df_candidates.index:
        comparisons[candidate_i] = {}
        for candidate_j in df_candidates.index:
            comparisons[candidate_i][candidate_j] = 0

    for i_pop in range(len(df_population)):
        ballot_candidates = df_population.iloc[i_pop][['choice_' + str(i_choice + 1) for i_choice in range(len(df_candidates))]].to_list()
        for i, candidate_i in enumerate(ballot_candidates[:-1]):
            for j, candidate_j in enumerate(ballot_candidates[i + 1:]):
                comparisons[candidate_i][candidate_j] += 1

    return comparisons


def get_x_zeroline():
    return go.Scatter(
        x=[-5, 5],
        y=[0, 0],
        mode='lines',
        line=dict(
            color="rgba(200, 212, 227, 0.25)",
        )
    )


def get_y_zeroline():
    return go.Scatter(
        x=[0, 0],
        y=[-5, 5],
        mode='lines',
        line=dict(
            color="rgba(200, 212, 227, 0.25)",
        )
    )


def plot_axes_and_population(
    population_dim_1_values,
    population_dim_2_values,
    save
):
    x_zeroline, y_zeroline = get_x_zeroline(), get_y_zeroline()
    population_data = go.Scatter(
        x=population_dim_1_values,
        y=population_dim_2_values,
        mode='markers',
        marker=dict(
            size=10,
            color='white',
            opacity=1/3
        )
    )

    axes_fig = go.Figure(
        layout=dict(
            template='plotly_dark',
            width=1080,
            height=1080,
            xaxis=dict(
                range=(-5, 5),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=(-5, 5),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        ),
        data = [x_zeroline, y_zeroline]
    )

    population_fig = go.Figure(
        layout=dict(
            template='plotly_dark',
            width=1080,
            height=1080,
            xaxis=dict(
                range=(-5, 5),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=(-5, 5),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        ),
        data = [population_data]
    )


    if save:
        axes_fig.write_image(f"../charts/axes.png")
        population_fig.write_image(f"../charts/population.png")
    else:
        axes_fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
        ))
        population_fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
        ))
        axes_fig.show()
        population_fig.show()


def plot_population(
    population_dim_1_values,
    population_dim_2_values,
    fps,
    save
):
    fig_dict = {
        'layout': {},
        'data': [],
        'frames': []
    }
    fig_dict['layout'] = dict(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 1000 / fps}}]
            )],
            visible=not save
        )],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    for i_pop in tqdm(range(N_POPULATION)):
        population_data = go.Scatter(
            x=population_dim_1_values[:i_pop + 1],
            y=population_dim_2_values[:i_pop + 1],
            mode='markers',
            marker=dict(
                size=10,
                color='white',
                opacity=1/3
            )
        )

        data = [population_data]

        if save:
            fig_dict["data"] = data
            fig = go.Figure(fig_dict)
            fig.write_image(f"../charts/population_plot/{i_pop}.png")

        else:
            fig_dict["frames"].append({
                "data": data
            })
            if len(fig_dict["data"]) == 0:
                fig_dict["data"] = data

    if not save:
        fig = go.Figure(fig_dict)
        fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
        ))
        fig.show()
        if electoral_system == "ranked_pairs":
            graph_fig = go.Figure(graph_fig_dict)
            graph_fig.update_layout(dict(
                paper_bgcolor='rgba(17, 17, 17, 1)',
                plot_bgcolor='rgba(17, 17, 17, 1)'
            ))
            graph_fig.show()


def plot_candidates(
    candidates_dim_1_values,
    candidates_dim_2_values,
    n_seconds_between_locs,
    fps,
    save
):
    fig_dict = {
        'layout': {},
        'data': [],
        'frames': []
    }
    fig_dict['layout'] = dict(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 1000 / fps}}]
            )],
            visible=not save
        )],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    candidates_pos_init_x = np.array([-3.5 + CANDIDATES.index(can) for can in CANDIDATES])
    candidates_pos_init_y = np.array([4 for _ in range(len(CANDIDATES))])

    i_image = 0
    for x_ in tqdm(np.arange(0, 1 + 1 / (fps * n_seconds_between_locs), 1 / (fps * n_seconds_between_locs))):
        x = ((1 - x_) * candidates_pos_init_x) + (x_ * candidates_dim_1_values[0])
        y = ((1 - x_) * candidates_pos_init_y) + (x_ * candidates_dim_2_values[0])
        candidates_data = go.Scatter(
            x=x,
            y=y,
            textfont_size=24,
            text=CANDIDATES,
            textposition='top center',
            mode='markers+text',
            marker=dict(
                size=50,
                color=COLORS,
                opacity=1,
                line={
                    'width': 2,
                    'color': 'rgb(17, 17, 17)',
                }
            )
        )

        data = [candidates_data]

        if save:
            fig_dict["data"] = data
            fig = go.Figure(fig_dict)
            fig.write_image(f"../charts/candidates_plot/{i_image}.png")

        else:
            fig_dict["frames"].append({
                "data": data
            })
            if len(fig_dict["data"]) == 0:
                fig_dict["data"] = data

        i_image += 1

    if not save:
        fig = go.Figure(fig_dict)
        fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
        ))
        fig.show()
        if electoral_system == "ranked_pairs":
            graph_fig = go.Figure(graph_fig_dict)
            graph_fig.update_layout(dict(
                paper_bgcolor='rgba(17, 17, 17, 1)',
                plot_bgcolor='rgba(17, 17, 17, 1)'
            ))
            graph_fig.show()


def plurality_vis(df_population, df_candidates):
    winners = get_winners_plurality(df_candidates)

    pie_plot = go.Pie(
        labels=df_candidates.index,
        values=df_candidates['choice_1'],
        textinfo='label',
        textfont=dict(
            color="white",
            size=48,
        ),
        textposition='inside',
        marker=dict(
            colors=COLORS,
            line=dict(
                color='rgb(17, 17, 17)',
                width=2,
            )
        ),
        pull=[1/10 if winners[0] == candidate else 0 for candidate in CANDIDATES],
        direction='clockwise',
        sort=False,
    )

    return [pie_plot]


def instant_runoff_vis(df_population, df_candidates):
    winners, round_votes = get_winners_instant_runoff(df_population, df_candidates)

    labels = [f'Round {round + 1}: {can}' for round in range(len(CANDIDATES)) for can in CANDIDATES]
    sources, targets, values = [], [], []
    link_colors = []
    for i_round, round_votes_ in enumerate(round_votes):
        if i_round == 0:
            continue
        loser = winners[len(winners) - i_round]
        for candidate in sorted(round_votes_.keys()):
            # if candidate in CANDIDATES[:4] and loser in CANDIDATES[:4]:
            sources.append((i_round - 1) * len(CANDIDATES) + CANDIDATES.index(candidate))
            targets.append(i_round * len(CANDIDATES) + CANDIDATES.index(candidate))
            values.append(round_votes[i_round - 1][candidate])
            link_colors.append(COLORS[CANDIDATES.index(candidate)].replace('rgb', 'rgba').replace(')', ', 0.33)'))

            if round_votes_[candidate] - round_votes[i_round - 1][candidate] > 0:
                sources.append((i_round - 1) * len(CANDIDATES) + CANDIDATES.index(loser))
                targets.append(i_round * len(CANDIDATES) + CANDIDATES.index(candidate))
                values.append(round_votes_[candidate] - round_votes[i_round - 1][candidate])
                link_colors.append(COLORS[CANDIDATES.index(candidate)].replace('rgb', 'rgba').replace(')', ', 0.33)'))

    x_start = 0
    x_end = 1
    y_start = 0
    y_end = 2
    candidate_space = []
    for candidates_in_round in range(len(CANDIDATES), 0, -1):
        d_y = (y_end - y_start) / (candidates_in_round + 1)
        y_ = y_start
        for _ in range(candidates_in_round):
            y_ += d_y
            candidate_space.append(y_)


    round_space = []
    d_x = (x_end - x_start) / (len(CANDIDATES) + 1)
    x_ = x_start
    for candidates_in_round in range(len(CANDIDATES), 0, -1):
        x_ += d_x
        for _ in range(candidates_in_round):
            round_space.append(x_)


    x, y = round_space, candidate_space
    sankey_plot = go.Sankey(
        domain=dict(
            x=[0, 1],
            y=[1/3, 2/3],
        ),
        ids=labels,
        node=dict(
            x=x,
            y=y,
            pad=0,
            thickness=48,
            line=dict(
                color="rgb(17, 17, 17)",
                width=0
            ),
            # label=labels,
            label=CANDIDATES * 8,
            color=COLORS * 8,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
        textfont=dict(
            size=24,
            color='white',
        ),
    )

    return [sankey_plot]


def ranked_pairs_vis(df_population, df_candidates):
    data = []

    candidate_coords = {
        "Alice": (1, (1 + np.sqrt(2))),
        "Bob": ((1 + np.sqrt(2)), 1),
        "Carol": ((1 + np.sqrt(2)), -1),
        "Dave": (1, -(1 + np.sqrt(2))),
        "Eve": (-1, -(1 + np.sqrt(2))),
        "Frank": (-(1 + np.sqrt(2)), -1),
        "Grace": (-(1 + np.sqrt(2)), 1),
        "Heidi": (-1, (1 + np.sqrt(2))),
    }

    comparisons = get_comparisons(df_population, df_candidates)
    for can_i in comparisons:
        for can_j in comparisons:
            if can_i != can_j and comparisons[can_i][can_j] > comparisons[can_j][can_i]:
                x0, y0 = candidate_coords[can_i]
                x1, y1 = candidate_coords[can_j]
                data.append(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(
                        width=1 + 25 * (comparisons[can_i][can_j] - 500) / 1000,
                        color=COLORS[CANDIDATES.index(can_i)],
                    )
                ))

    node_x = []
    node_y = []
    for candidate in candidate_coords:
        x, y = candidate_coords[candidate]
        node_x.append(x)
        node_y.append(y)

    winners = get_winners_ranked_pairs(df_population, df_candidates, comparisons=comparisons)

    data.append(go.Scatter(
        x=node_x,
        y=node_y,
        textfont_size=24,
        text=CANDIDATES,
        mode='markers+text',
        marker_symbol=['star' if candidate == winners[0] else 'circle' for candidate in CANDIDATES],
        marker=dict(
            size=100,
            color=COLORS,
            line={
                'width': 5,
                'color': 'rgb(17, 17, 17)',
            }
        )
    ))

    return data


def candidate_random_movement(
    electoral_system,
    population_dim_1_values,
    population_dim_2_values,
    candidates_dim_1_values,
    candidates_dim_2_values,
    n_samples,
    n_seconds_between_locs,
    fps,
    save=False
):

    df_population, df_candidates = initialize_dfs(
        candidates_dim_1_values[0],
        candidates_dim_2_values[0],
        population_dim_1_values,
        population_dim_2_values
    )

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"] = go.Layout(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 1000 / fps}}]
            )],
            visible=not save
        )],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    electoral_system_fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    electoral_system_fig_dict["layout"] = go.Layout(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-3, 3),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-3, 3),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 1000 / fps}}]
            )],
            visible=not save
        )],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


    i_image = 0
    for i_sample in tqdm(range(n_samples)):
        candidate_dim_1_init = candidates_dim_1_values[i_sample]
        candidate_dim_2_init = candidates_dim_2_values[i_sample]
        if i_sample < n_samples - 1:
            candidate_dim_1_final = candidates_dim_1_values[i_sample + 1]
            candidate_dim_2_final = candidates_dim_2_values[i_sample + 1]
        else:
            candidate_dim_1_final = candidates_dim_1_values[0]
            candidate_dim_2_final = candidates_dim_2_values[0]

        for x in tqdm(np.arange(0, 1, 1 / (n_seconds_between_locs * fps))):
            df_candidates["dim_1"] = ((1 - x) * candidate_dim_1_init) + (x * candidate_dim_1_final)
            df_candidates["dim_2"] = ((1 - x) * candidate_dim_2_init) + (x * candidate_dim_2_final)
            df_population, df_candidates = update_rankings(df_population, df_candidates)
            winners = get_winners(df_population, df_candidates, electoral_system)

            candidates_plot = go.Scatter(
                x=df_candidates['dim_1'],
                y=df_candidates['dim_2'],
                textfont_size=24,
                text=CANDIDATES,
                textposition='top center',
                mode='markers+text',
                marker_symbol=['star' if candidate == winners[0] else 'circle' for candidate in CANDIDATES],
                marker=dict(
                    # size=[50 - (winners.index(candidate) * 5) for candidate in CANDIDATES],
                    size=50,
                    color=COLORS,
                    opacity=1,
                    line={
                        'width': 2,
                        'color': 'rgb(17, 17, 17)',
                    }
                )
            )

            population_plot = go.Scatter(
                x=df_population['dim_1'],
                y=df_population['dim_2'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[COLORS[CANDIDATES.index(df_population['choice_1'].iloc[i_pop])] for i_pop in range(N_POPULATION)],
                    opacity=1/3
                )
            )

            data = [population_plot, candidates_plot]

            if electoral_system == "plurality":
                electoral_system_data = plurality_vis(df_population, df_candidates)
            elif electoral_system == "instant_runoff":
                electoral_system_data = instant_runoff_vis(df_population, df_candidates)
            elif electoral_system == "ranked_pairs":
                electoral_system_data = ranked_pairs_vis(df_population, df_candidates)

            if save:
                fig_dict["data"] = data
                fig = go.Figure(fig_dict)
                fig.write_image(f"../charts/random_movement/{electoral_system}/{i_image}.png")

                electoral_system_fig_dict["data"] = electoral_system_data
                electoral_system_fig = go.Figure(electoral_system_fig_dict)
                electoral_system_fig.write_image(f"../charts/random_movement/electoral_system_visualizations/{electoral_system}/{i_image}.png")
            else:
                fig_dict["frames"].append({
                    "data": data
                })
                if len(fig_dict["data"]) == 0:
                    fig_dict["data"] = data

                electoral_system_fig_dict["frames"].append({
                    "data": electoral_system_data
                })
                if len(electoral_system_fig_dict["data"]) == 0:
                    electoral_system_fig_dict["data"] = electoral_system_data

            i_image += 1

    if not save:
        fig = go.Figure(fig_dict)
        fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
        ))
        fig.show()

        electoral_system_fig = go.Figure(electoral_system_fig_dict)
        electoral_system_fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
        ))
        electoral_system_fig.show()


def partisan_random_movement(
    electoral_system,
    population_dim_1_values,
    population_dim_2_values,
    n_samples,
    n_seconds_between_locs,
    fps,
    save=False
):
    candidates_dim_1_values = np.concatenate([
        -1 * abs(np.random.normal(0, 1, (n_samples, len(CANDIDATES) // 2))),
        abs(np.random.normal(0, 1, (n_samples, len(CANDIDATES) // 2)))
    ], axis=1)
    candidates_dim_2_values = np.random.normal(0, 1, (n_samples, len(CANDIDATES)))

    df_population, df_candidates = initialize_dfs_partisan(
        candidates_dim_1_values[0],
        candidates_dim_2_values[0],
        population_dim_1_values,
        population_dim_2_values
    )

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"] = go.Layout(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 1000 / fps}}]
            )],
            visible=not save
        )],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    if electoral_system == "ranked_pairs":
        graph_fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }
        graph_fig_dict["layout"] = go.Layout(
            template='plotly_dark',
            width=1080,
            height=1080,
            xaxis=dict(
                range=(-3, 3),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=(-3, 3),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            showlegend=False,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 1000 / fps}}]
                )],
                visible=not save
            )],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

    x_zeroline, y_zeroline = get_x_zeroline(), get_y_zeroline()

    i_image = 0
    for i_sample in tqdm(range(n_samples)):
        candidate_dim_1_init = candidates_dim_1_values[i_sample]
        candidate_dim_2_init = candidates_dim_2_values[i_sample]
        if i_sample < n_samples - 1:
            candidate_dim_1_final = candidates_dim_1_values[i_sample + 1]
            candidate_dim_2_final = candidates_dim_2_values[i_sample + 1]
        else:
            candidate_dim_1_final = candidates_dim_1_values[0]
            candidate_dim_2_final = candidates_dim_2_values[0]

        for x in tqdm(np.arange(0, 1, 1 / (n_seconds_between_locs * fps))):
            df_candidates["dim_1"] = ((1 - x) * candidate_dim_1_init) + (x * candidate_dim_1_final)
            df_candidates["dim_2"] = ((1 - x) * candidate_dim_2_init) + (x * candidate_dim_2_final)
            df_population, df_candidates = update_rankings_partisan(df_population, df_candidates)
            left_winners = get_winners(df_population, df_candidates.iloc[:len(CANDIDATES) // 2], electoral_system)
            right_winners = get_winners(df_population, df_candidates.iloc[len(CANDIDATES) // 2:], electoral_system)

            left_candidates_plot = go.Scatter(
                x=df_candidates['dim_1'][df_candidates["dim_1"] < 0],
                y=df_candidates['dim_2'][df_candidates["dim_1"] < 0],
                text=CANDIDATES[:len(CANDIDATES) // 2],
                textposition='top center',
                mode='markers+text',
                marker_symbol=['star' if candidate == left_winners[0] else 'circle' for candidate in CANDIDATES[:len(CANDIDATES) // 2]],
                marker=dict(
                    # size=[50 - (left_winners.index(candidate) * 5) for candidate in CANDIDATES[:len(CANDIDATES) // 2]],
                    size=50,
                    color=LEFT_COLORS,
                    opacity=1,
                )
            )

            right_candidates_plot = go.Scatter(
                x=df_candidates['dim_1'][df_candidates["dim_1"] > 0],
                y=df_candidates['dim_2'][df_candidates["dim_1"] > 0],
                text=CANDIDATES[len(CANDIDATES) // 2:],
                textposition='top center',
                mode='markers+text',
                marker_symbol=['star' if candidate == right_winners[0] else 'circle' for candidate in CANDIDATES[len(CANDIDATES) // 2:]],
                marker=dict(
                    # size=[50 - (right_winners.index(candidate) * 5) for candidate in CANDIDATES[len(CANDIDATES) // 2:]],
                    size=50,
                    color=RIGHT_COLORS,
                    opacity=1,
                )
            )

            left_population_plot = go.Scatter(
                x=df_population['dim_1'][df_population['dim_1'] < 0],
                y=df_population['dim_2'][df_population['dim_1'] < 0],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[LEFT_COLORS[CANDIDATES.index(df_population[df_population['dim_1'] < 0]['choice_1'].iloc[i_pop])] for i_pop in range(len(df_population[df_population['dim_1'] < 0]))],
                    opacity=1/3
                )
            )

            right_population_plot = go.Scatter(
                x=df_population['dim_1'][df_population['dim_1'] > 0],
                y=df_population['dim_2'][df_population['dim_1'] > 0],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[RIGHT_COLORS[CANDIDATES[len(CANDIDATES) // 2:].index(df_population[df_population['dim_1'] > 0]['choice_1'].iloc[i_pop])] for i_pop in range(len(df_population[df_population['dim_1'] > 0]))],
                    opacity=1/3
                )
            )

            data = [left_candidates_plot, right_candidates_plot, left_population_plot, right_population_plot, x_zeroline, y_zeroline]

            if electoral_system == "ranked_pairs":
                graph_data = ranked_pairs_vis(df_population)

            if save:
                fig_dict["data"] = data
                fig = go.Figure(fig_dict)
                fig.write_image(f"../charts/random_movement/partisan/{electoral_system}/{i_image}.png")
                if electoral_system == "ranked_pairs":
                    graph_fig_dict["data"] = graph_data
                    graph_fig = go.Figure(graph_fig_dict)
                    graph_fig.write_image(f"../charts/random_movement/graph/partisan/{i_image}.png")

            else:
                fig_dict["frames"].append({
                    "data": data
                })
                if len(fig_dict["data"]) == 0:
                    fig_dict["data"] = data

                if electoral_system == "ranked_pairs":
                    graph_fig_dict["frames"].append({
                        "data": graph_data
                    })
                    if len(graph_fig_dict["data"]) == 0:
                        graph_fig_dict["data"] = graph_data

            i_image += 1

    if not save:
        fig = go.Figure(fig_dict)
        fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
        ))
        fig.show()
        if electoral_system == "ranked_pairs":
            graph_fig = go.Figure(graph_fig_dict)
            graph_fig.update_layout(dict(
                paper_bgcolor='rgba(17, 17, 17, 1)',
                plot_bgcolor='rgba(17, 17, 17, 1)'
            ))
            graph_fig.show()


def generate_partisan_winners(
    electoral_system,
    population_dim_1_values,
    population_dim_2_values,
    n_samples,
    save=False
):
    candidates_dim_1_values = np.concatenate([
        -1 * abs(np.random.normal(0, 1, (n_samples, len(CANDIDATES) // 2))),
        abs(np.random.normal(0, 1, (n_samples, len(CANDIDATES) // 2)))
    ], axis=1)
    candidates_dim_2_values = np.random.normal(0, 1, (n_samples, len(CANDIDATES)))

    left_winners_dim_1 = []
    left_winners_dim_2 = []
    right_winners_dim_1 = []
    right_winners_dim_2 = []
    general_winners_dim_1 = []
    general_winners_dim_2 = []
    for i_sample in tqdm(range(n_samples)):
        df_population, df_candidates = initialize_dfs_partisan(
            candidates_dim_1_values[i_sample],
            candidates_dim_2_values[i_sample],
            population_dim_1_values,
            population_dim_2_values
        )
        df_population, df_candidates = update_rankings_partisan(df_population, df_candidates)
        left_winners = get_winners(df_population[df_population['dim_1'] < 0], df_candidates.iloc[:len(CANDIDATES) // 2], electoral_system)
        right_winners = get_winners(df_population[df_population['dim_1'] > 0], df_candidates.iloc[len(CANDIDATES) // 2:], electoral_system)
        general_candidates = [left_winners[0], right_winners[0]]

        population_dims = np.array([population_dim_1_values, population_dim_2_values]).T.reshape(N_POPULATION, 1, 2)
        candidate_dims = np.array([df_candidates[['dim_1']].loc[general_candidates].values, df_candidates[['dim_2']].loc[general_candidates].values]).T.reshape(1, 2, 2)
        euclidean_distances = np.sqrt(np.sum(np.power((population_dims - candidate_dims), 2), axis=2))
        rankings = np.argsort(euclidean_distances)

        left_winner_dim_1, left_winner_dim_2 = df_candidates[['dim_1', 'dim_2']].loc[left_winners[0]]
        right_winner_dim_1, right_winner_dim_2 = df_candidates[['dim_1', 'dim_2']].loc[right_winners[0]]

        if (rankings[:, 0] == 0).sum() > N_POPULATION / 2:
            general_winner_dim_1 = left_winner_dim_1
            general_winner_dim_2 = left_winner_dim_2
        else:
            general_winner_dim_1 = right_winner_dim_1
            general_winner_dim_2 = right_winner_dim_2

        left_winners_dim_1.append(left_winner_dim_1)
        left_winners_dim_2.append(left_winner_dim_2)
        right_winners_dim_1.append(right_winner_dim_1)
        right_winners_dim_2.append(right_winner_dim_2)
        general_winners_dim_1.append(general_winner_dim_1)
        general_winners_dim_2.append(general_winner_dim_2)

    if save:
        pickle.dump({
            'left_dim_1': left_winners_dim_1,
            'left_dim_2': left_winners_dim_2,
            'right_dim_1': right_winners_dim_1,
            'right_dim_2': right_winners_dim_2,
            'general_dim_1': general_winners_dim_1,
            'general_dim_2': general_winners_dim_2
        }, open(f'winners/partisan_{electoral_system}.p', 'wb'))


def plot_partisan_winner_space(
    electoral_system,
    save=False
):
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"] = go.Layout(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    partisan_winners_dims = pickle.load(open(f"winners/partisan_{electoral_system}.p", "rb"))

    # for i_sample in tqdm(range(n_samples)):
    #     df_population, df_candidates = initialize_dfs(
    #         candidates_dim_1_values[i_sample],
    #         candidates_dim_2_values[i_sample],
    #         population_dim_1_values,
    #         population_dim_2_values
    #     )
        # winners = get_winners(df_population, df_candidates, electoral_system)
        # winner_dim_1, winner_dim_2 = df_candidates[['dim_1', 'dim_2']].loc[winners[0]]
        # winners_dim_1.append(winner_dim_1)
        # winners_dim_2.append(winner_dim_2)

        # winners_dim_1 = winners_dim_1[-n_samples_per_kde:]
        # winners_dim_2 = winners_dim_2[-n_samples_per_kde:]
        # if len(winners_dim_1) < n_samples_per_kde:
        #     continue

        # population_plot = go.Scatter(
        #     x=df_population['dim_1'],
        #     y=df_population['dim_2'],
        #     mode='markers',
        #     marker=dict(
        #         size=10,
        #         color='white',
        #         opacity=1/10
        #     )
        # )

    # build a gaussian kde from the winners, and evaluate on a regular grid
    # x_dim = partisan_winners_dims['left_dim_1'] + partisan_winners_dims['right_dim_1']
    # y_dim = partisan_winners_dims['left_dim_2'] + partisan_winners_dims['right_dim_2']
    x_dim = partisan_winners_dims['general_dim_1']
    y_dim = partisan_winners_dims['general_dim_2']
    kde = gaussian_kde(np.vstack([x_dim, y_dim]))
    xgrid = np.linspace(-5, 5, 100)
    ygrid = np.linspace(-5, 5, 100)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    winner_space_contour_plot = go.Heatmap(
        z=Z.reshape(Xgrid.shape),
        x0=-5,
        dx=0.1,
        y0=-5,
        dy=0.1,
        # colorscale=[(0, "rgb(17, 17, 17)"), (0.2, "rgb(17, 17, 17)"), (1, "rgb(93, 105, 177)")],
        colorscale=[(0, "rgba(93, 105, 177, 0)"), (0.15, "rgba(93, 105, 177, 0)"), (0.55, "rgba(80, 175, 199, 0.33)"), (1, "rgba(255, 255, 255, 0.67)")],
        # contours_showlines=False,
        zsmooth='best',
        showscale=False
    )

    # winners_plot = go.Scatter(
    #     x=winners_dims['dim_1'],
    #     y=winners_dims['dim_2'],
    #     mode='markers',
    #     marker=dict(
    #         size=10,
    #         color="rgba(93, 105, 177, 0.25)"
    #     )
    # )

    x_zeroline, y_zeroline = get_x_zeroline(), get_y_zeroline()

    # data = [population_plot, winner_space_contour_plot, x_zeroline, y_zeroline]
    # data = [population_plot, x_zeroline, y_zeroline]
    data = [winner_space_contour_plot]
    # data = [winners_plot, x_zeroline, y_zeroline]

    if save:
        fig_dict["data"] = data
        fig = go.Figure(fig_dict)
        fig.write_image(f"../charts/winner_space/partisan_{electoral_system}.png")
    else:
        fig_dict["frames"].append({
            "data": data
        })
        if len(fig_dict["data"]) == 0:
            fig_dict["data"] = data

    # if save:
    #     fig.write_image(f'../charts/winner_space/{electoral_system}.png')

    if not save:
        fig = go.Figure(fig_dict)
        fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
            # paper_bgcolor='rgba(0, 0, 0, 0)',
            # plot_bgcolor='rgba(0, 0, 0, 0)'
        ))
        fig.show()


def generate_winners(
    electoral_system,
    population_dim_1_values,
    population_dim_2_values,
    candidates_dim_1_values,
    candidates_dim_2_values,
    n_samples,
    save=False
):

    winners_dim_1 = []
    winners_dim_2 = []
    for i_sample in tqdm(range(n_samples)):
        df_population, df_candidates = initialize_dfs(
            candidates_dim_1_values[i_sample],
            candidates_dim_2_values[i_sample],
            population_dim_1_values,
            population_dim_2_values
        )

        winners = get_winners(df_population, df_candidates, electoral_system)
        winner_dim_1, winner_dim_2 = df_candidates[['dim_1', 'dim_2']].loc[winners[0]]
        winners_dim_1.append(winner_dim_1)
        winners_dim_2.append(winner_dim_2)

    if save:
        pickle.dump({
            'dim_1': winners_dim_1,
            'dim_2': winners_dim_2
        }, open(f'winners/{electoral_system}.p', 'wb'))


def plot_winner_space(
    electoral_system,
    save=False
):
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"] = go.Layout(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    winners_dims = pickle.load(open(f"winners/{electoral_system}.p", "rb"))

    # for i_sample in tqdm(range(n_samples)):
    #     df_population, df_candidates = initialize_dfs(
    #         candidates_dim_1_values[i_sample],
    #         candidates_dim_2_values[i_sample],
    #         population_dim_1_values,
    #         population_dim_2_values
    #     )
        # winners = get_winners(df_population, df_candidates, electoral_system)
        # winner_dim_1, winner_dim_2 = df_candidates[['dim_1', 'dim_2']].loc[winners[0]]
        # winners_dim_1.append(winner_dim_1)
        # winners_dim_2.append(winner_dim_2)

        # winners_dim_1 = winners_dim_1[-n_samples_per_kde:]
        # winners_dim_2 = winners_dim_2[-n_samples_per_kde:]
        # if len(winners_dim_1) < n_samples_per_kde:
        #     continue

        # population_plot = go.Scatter(
        #     x=df_population['dim_1'],
        #     y=df_population['dim_2'],
        #     mode='markers',
        #     marker=dict(
        #         size=10,
        #         color='white',
        #         opacity=1/10
        #     )
        # )

    # build a gaussian kde from the winners, and evaluate on a regular grid
    kde = gaussian_kde(np.vstack([winners_dims['dim_1'], winners_dims['dim_2']]))
    xgrid = np.linspace(-5, 5, 100)
    ygrid = np.linspace(-5, 5, 100)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    winner_space_contour_plot = go.Heatmap(
        z=Z.reshape(Xgrid.shape),
        x0=-5,
        dx=0.1,
        y0=-5,
        dy=0.1,
        # colorscale=[(0, "rgb(17, 17, 17)"), (0.2, "rgb(17, 17, 17)"), (1, "rgb(93, 105, 177)")],
        colorscale=[(0, "rgba(93, 105, 177, 0)"), (0.15, "rgba(93, 105, 177, 0)"), (0.55, "rgba(80, 175, 199, 0.33)"), (1, "rgba(255, 255, 255, 0.67)")],
        # contours_showlines=False,
        zsmooth='best',
        showscale=False
    )

    winners_plot = go.Scatter(
        x=winners_dims['dim_1'],
        y=winners_dims['dim_2'],
        mode='markers',
        marker=dict(
            size=10,
            color="rgba(93, 105, 177, 0.25)"
        )
    )

    x_zeroline, y_zeroline = get_x_zeroline(), get_y_zeroline()

    # data = [population_plot, winner_space_contour_plot, x_zeroline, y_zeroline]
    # data = [population_plot, x_zeroline, y_zeroline]
    data = [winner_space_contour_plot]
    # data = [winners_plot, x_zeroline, y_zeroline]

    if save:
        fig_dict["data"] = data
        fig = go.Figure(fig_dict)
        fig.write_image(f"../charts/winner_space/{electoral_system}.png")
    else:
        fig_dict["frames"].append({
            "data": data
        })
        if len(fig_dict["data"]) == 0:
            fig_dict["data"] = data

    # if save:
    #     fig.write_image(f'../charts/winner_space/{electoral_system}.png')

    if not save:
        fig = go.Figure(fig_dict)
        fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
            # paper_bgcolor='rgba(0, 0, 0, 0)',
            # plot_bgcolor='rgba(0, 0, 0, 0)'
        ))
        fig.show()


def plot_winner_space_transition(
    from_electoral_system,
    to_electoral_system,
    n_seconds_between_locs,
    fps,
    save=False
):
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    fig_dict["layout"] = go.Layout(
        template='plotly_dark',
        width=1080,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 1000 / fps}}]
            )],
            visible=not save
        )],
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    xgrid = np.linspace(-5, 5, 100)
    ygrid = np.linspace(-5, 5, 100)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    winners_dims_from = pickle.load(open(f"winners/{from_electoral_system}.p", "rb"))
    winners_dims_to = pickle.load(open(f"winners/{to_electoral_system}.p", "rb"))

    kde_from = gaussian_kde(np.vstack([winners_dims_from['dim_1'], winners_dims_from['dim_2']]))
    kde_to = gaussian_kde(np.vstack([winners_dims_to['dim_1'], winners_dims_to['dim_2']]))
    Z_from = kde_from.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    Z_to = kde_to.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    i_sample = 0
    for x in tqdm(np.arange(0, 1, 1 / (fps * n_seconds_between_locs))):

        Z_sample = (1 - x) * Z_from + x * Z_to

        winner_space_contour_plot = go.Heatmap(
            z=Z_sample.reshape(Xgrid.shape),
            x0=-5,
            dx=0.1,
            y0=-5,
            dy=0.1,
            # colorscale=[(0, "rgb(17, 17, 17)"), (0.2, "rgb(17, 17, 17)"), (1, "rgb(93, 105, 177)")],
            colorscale=[(0, "rgba(93, 105, 177, 0)"), (0.15, "rgba(93, 105, 177, 0)"), (0.55, "rgba(80, 175, 199, 0.5)"), (1, "rgba(255, 255, 255, 1)")],
            # contours_showlines=False,
            zsmooth='best',
            showscale=False
        )

        data = [winner_space_contour_plot]

        if save:
            fig_dict["data"] = data
            fig = go.Figure(fig_dict)
            fig.write_image(f"../charts/winner_space_transition/{from_electoral_system}_{to_electoral_system}/{i_sample}.png")
        else:
            fig_dict["frames"].append({
                "data": data
            })
            if len(fig_dict["data"]) == 0:
                fig_dict["data"] = data

        i_sample += 1

    if not save:
        fig = go.Figure(fig_dict)
        fig.update_layout(dict(
            paper_bgcolor='rgba(17, 17, 17, 1)',
            plot_bgcolor='rgba(17, 17, 17, 1)'
            # paper_bgcolor='rgba(0, 0, 0, 0)',
            # plot_bgcolor='rgba(0, 0, 0, 0)'
        ))
        fig.show()


if __name__ == '__main__':
    save = 0
    n_samples = 10 if save else 1
    n_seconds_between_locs = 5 if save else 1
    fps = 5 if save else 1

    population_dim_1_values = np.random.normal(0, 1, N_POPULATION)
    population_dim_2_values = np.random.normal(0, 1, N_POPULATION)
    candidates_dim_1_values = np.random.normal(0, 1, (n_samples, len(CANDIDATES)))
    candidates_dim_2_values = np.random.normal(0, 1, (n_samples, len(CANDIDATES)))

    # plot_population(
    #     population_dim_1_values,
    #     population_dim_2_values,
    #     fps,
    #     save
    # )

    # plot_candidates(
    #     candidates_dim_1_values,
    #     candidates_dim_2_values,
    #     n_seconds_between_locs,
    #     fps,
    #     save
    # )

    # plot_axes_and_population(
    #     population_dim_1_values,
    #     population_dim_2_values,
    #     save
    # )

    # electoral_system = "plurality"
    # electoral_system = "instant_runoff"
    # electoral_system = "ranked_pairs"
    for electoral_system in ["plurality", "instant_runoff", "ranked_pairs"]:
        # generate_winners(
        #     electoral_system,
        #     population_dim_1_values,
        #     population_dim_2_values,
        #     candidates_dim_1_values,
        #     candidates_dim_2_values,
        #     n_samples,
        #     save
        # )
        # plot_winner_space(
        #     electoral_system,
        #     save
        # )
        # generate_partisan_winners(
        #     electoral_system,
        #     population_dim_1_values,
        #     population_dim_2_values,
        #     n_samples,
        #     save
        # )
        # plot_partisan_winner_space(
        #     electoral_system,
        #     save
        # )
        candidate_random_movement(
            electoral_system,
            population_dim_1_values,
            population_dim_2_values,
            candidates_dim_1_values,
            candidates_dim_2_values,
            n_samples,
            n_seconds_between_locs,
            fps,
            save
        )
        # partisan_random_movement(
        #     electoral_system,
        #     population_dim_1_values,
        #     population_dim_2_values,
        #     n_samples,
        #     n_seconds_between_locs,
        #     fps,
        #     save
        # )

    # from_electoral_system = "plurality"
    # to_electoral_system = "instant_runoff"
    # plot_winner_space_transition(
    #     from_electoral_system,
    #     to_electoral_system,
    #     n_seconds_between_locs,
    #     fps,
    #     save
    # )
