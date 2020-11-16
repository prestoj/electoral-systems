import voting_systems as vs
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde


np.random.seed(1)

CANDIDATES = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Heidi']
N_POPULATION = 1000
COLORS = px.colors.qualitative.Vivid[:len(CANDIDATES)]


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


def initialize_dfs(
    candidate_dim_1_values,
    candidate_dim_2_values,
    population_dim_1_values,
    population_dim_2_values
):
    df_population  = pd.DataFrame({
        'dim_1': population_dim_1_values,
        'dim_2': population_dim_2_values,
        'favorite': 'neutral',
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


def get_winners(df_population, df_candidates, electoral_system):
    if electoral_system == "plurality":
        return get_winners_plurality(df_candidates)
    elif electoral_system == "instant_runoff":
        return get_winners_instant_runoff(df_population, df_candidates)
    elif electoral_system == "ranked_pairs":
        return get_winners_ranked_pairs(df_population)
    elif electoral_system == "borda":
        return get_winners_borda(df_population, df_candidates)


def get_winners_plurality(df_candidates):
    return df_candidates.index[df_candidates['choice_1'].argsort().to_list()].to_list()[::-1]


def get_winners_instant_runoff(df_population, df_candidates):
    ballots = {}
    for i_pop in range(N_POPULATION):
        ballot = ""
        for i_choice in range(len(CANDIDATES)):
            ballot += df_population['choice_' + str(i_choice + 1)].iloc[i_pop]
            if i_choice != len(CANDIDATES):
                ballot += '-'

        if ballot in ballots.keys():
            ballots[ballot] += 1
        else:
            ballots[ballot] = 1

    return vs.rank_instant_runoff(ballots, N_POPULATION)[0]


def get_winners_ranked_pairs(df_population, comparisons=None):
    if comparisons is None:
        comparisons = get_comparisons(df_population)

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


def get_comparisons(df_population):
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

    return comparisons


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
        )]
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
            )]
        )

    # x zeroline
    x_zeroline = go.Scatter(
        x=[-5, 5],
        y=[0, 0],
        mode='lines',
        line=dict(
            color="rgba(200, 212, 227, 0.25)",
        )
    )

    # y zeroline
    y_zeroline = go.Scatter(
        x=[0, 0],
        y=[-5, 5],
        mode='lines',
        line=dict(
            color="rgba(200, 212, 227, 0.25)",
        )
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
                text=CANDIDATES,
                textposition='top center',
                mode='markers+text',
                marker_symbol=['star' if candidate == winners[0] else 'circle' for candidate in CANDIDATES],
                marker=dict(
                    size=[50 - (winners.index(candidate) * 5) for candidate in CANDIDATES],
                    color=COLORS,
                    opacity=1,
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

            data = [population_plot, candidates_plot, x_zeroline, y_zeroline]

            if electoral_system == "ranked_pairs":
                graph_data = graph_network_data(df_population)

            if save:
                fig_dict["data"] = data
                fig = go.Figure(fig_dict)
                fig.write_image(f"../charts/random_movement/{electoral_system}/{i_image}.png")
                if electoral_system == "ranked_pairs":
                    graph_fig_dict["data"] = graph_data
                    graph_fig = go.Figure(graph_fig_dict)
                    graph_fig.write_image(f"../charts/random_movement/graph/{i_image}.png")

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
        fig.show()
        if electoral_system == "ranked_pairs":
            graph_fig = go.Figure(graph_fig_dict)
            graph_fig.show()


def graph_network_data(df_population):
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

    comparisons = get_comparisons(df_population)
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

    winners = get_winners_ranked_pairs(df_population, comparisons)

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


def winner_space(
    electoral_system,
    population_dim_1_values,
    population_dim_2_values,
    candidates_dim_1_values,
    candidates_dim_2_values,
    n_samples,
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
    )

    n_samples_per_kde = 250

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

        winners_dim_1 = winners_dim_1[-n_samples_per_kde:]
        winners_dim_2 = winners_dim_2[-n_samples_per_kde:]
        if len(winners_dim_1) < n_samples_per_kde:
            continue

        population_plot = go.Scatter(
            x=df_population['dim_1'],
            y=df_population['dim_2'],
            mode='markers',
            marker=dict(
                size=10,
                color='white',
                opacity=1/10
            )
        )

        # build a gaussian kde from the winners, and evaluate on a regular grid
        kde = gaussian_kde(np.vstack([winners_dim_1, winners_dim_2]))
        xgrid = np.linspace(-5, 5, 100)
        ygrid = np.linspace(-5, 5, 100)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

        winner_space_contour_plot = go.Contour(
            z=Z.reshape(Xgrid.shape),
            x0=-5,
            dx=0.1,
            y0=-5,
            dy=0.1,
            colorscale=[(0, "rgb(17, 17, 17)"), (0.2, "rgb(17, 17, 17)"), (1, "rgb(93, 105, 177)")],
            contours_showlines=False,
            showscale=False
        )

        x_zeroline = go.Scatter(
            x=[-5, 5],
            y=[0, 0],
            mode='lines',
            line=dict(
                color="rgba(200, 212, 227, 0.25)",
            )
        )

        y_zeroline = go.Scatter(
            x=[0, 0],
            y=[-5, 5],
            mode='lines',
            line=dict(
                color="rgba(200, 212, 227, 0.25)",
            )
        )

        data = [population_plot, winner_space_contour_plot, x_zeroline, y_zeroline]

        if save:
            fig_dict["data"] = data
            fig = go.Figure(fig_dict)
            fig.write_image(f"../charts/winner_space/{electoral_system}/{i_sample}.png")
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
        fig.show()


if __name__ == '__main__':
    save = 1
    n_samples = 1000
    n_seconds_between_locs = 5
    fps = 30 if save else 10

    population_dim_1_values = np.random.normal(0, 1, N_POPULATION)
    population_dim_2_values = np.random.normal(0, 1, N_POPULATION)
    candidates_dim_1_values = np.random.normal(0, 1, (n_samples, len(CANDIDATES)))
    candidates_dim_2_values = np.random.normal(0, 1, (n_samples, len(CANDIDATES)))

    # electoral_system = "plurality"
    # electoral_system = "instant_runoff"
    # electoral_system = "ranked_pairs"
    for electoral_system in ["plurality", "instant_runoff", "ranked_pairs"]:
        winner_space(
            electoral_system,
            population_dim_1_values,
            population_dim_2_values,
            candidates_dim_1_values,
            candidates_dim_2_values,
            n_samples,
            save
        )
        # candidate_random_movement(
        #     electoral_system,
        #     population_dim_1_values,
        #     population_dim_2_values,
        #     candidates_dim_1_values,
        #     candidates_dim_2_values,
        #     n_samples,
        #     n_seconds_between_locs,
        #     fps,
        #     save
        # )

    # print(pio.templates['plotly_dark'])
