import voting_systems as vs
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from tqdm import tqdm

np.random.seed(0)

IS_RIGHT = 0


LEFT = [0, 1, 2, 3, 4, 5, 6, 7]
RIGHT = [4, 5, 6, 7]

CANDIDATES = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Heidi']

CANDIDATES = [CANDIDATES[i] for i in (RIGHT if IS_RIGHT else LEFT)]
N_POPULATION = 1000 // 2
COOL_COLORS = ['BLUE', 'TEAL', 'GREEN', 'CYAN']
WARM_COLORS = ['SALMON', 'ORANGE', 'YELLOW', 'PINK']

# COLORS = WARM_COLORS if IS_RIGHT else COOL_COLORS
COLORS = COOL_COLORS + WARM_COLORS


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


def initialize_dfs(dim_1_values, dim_2_values):
    df_population  = pd.DataFrame({
        'dim_1': np.abs(np.random.normal(0, 1, N_POPULATION)) * (1 if IS_RIGHT else -1),
        'dim_2': np.random.normal(0, 1, N_POPULATION),
        'favorite': 'neutral',
    })

    df_candidates  = pd.DataFrame({
        'dim_1': dim_1_values,
        'dim_2': dim_2_values,
        'mean_distance': -1,
    }, index=CANDIDATES)

    for i_candidate in range(len(CANDIDATES)):
        df_population['choice_' + str(i_candidate + 1)] = -1
        df_population['distance_' + str(i_candidate + 1)] = -1
        df_candidates['choice_' + str(i_candidate + 1)] = -1

    return update_rankings(df_population, df_candidates)


def candidate_random_movement():
    n_locs = 10
    n_seconds_between_locs = 5
    fps = 10
    candidates_dim_1_values = np.abs(np.random.normal(0, 1, (n_locs, len(CANDIDATES)))) * (1 if IS_RIGHT else -1)
    candidates_dim_2_values = np.random.normal(0, 1, (n_locs, len(CANDIDATES)))

    df_population, df_candidates = initialize_dfs(candidates_dim_1_values[0], candidates_dim_2_values[0])

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"] = dict(
        template='plotly_dark',
        width=1920,
        height=1080,
        xaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=(-5, 5),
            showgrid=False,
            showticklabels=False,
        ),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 1000 / fps}}]
            )]
        )]
    )

    i_image = 0
    for i_loc in tqdm(range(n_locs)):
        candidate_dim_1_init = candidates_dim_1_values[i_loc]
        candidate_dim_2_init = candidates_dim_2_values[i_loc]
        if i_loc < n_locs - 1:
            candidate_dim_1_final = candidates_dim_1_values[i_loc + 1]
            candidate_dim_2_final = candidates_dim_2_values[i_loc + 1]
        else:
            candidate_dim_1_final = candidates_dim_1_values[0]
            candidate_dim_2_final = candidates_dim_2_values[0]

        for x in tqdm(np.arange(0, 1, 1 / (n_seconds_between_locs * fps))):
            df_candidates["dim_1"] = ((1 - x) * candidate_dim_1_init) + (x * candidate_dim_1_final)
            df_candidates["dim_2"] = ((1 - x) * candidate_dim_2_init) + (x * candidate_dim_2_final)
            df_population, df_candidates = update_rankings(df_population, df_candidates)

            candidates_plot = go.Scatter(
                x=df_candidates['dim_1'],
                y=df_candidates['dim_2'],
                text=CANDIDATES,
                textposition='top center',
                mode='markers+text',
                marker_symbol=['star' if i_candidate == df_candidates['choice_1'].argmax() else 'circle' for i_candidate in range(len(CANDIDATES))],
                marker=dict(
                    size=10 + 40 * (df_candidates['choice_1'] / df_candidates['choice_1'].max()),
                    color=COLORS,
                    opacity=1
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

            fig_dict["frames"].append({
                "data": [population_plot, candidates_plot]
            })
            if x == 0 and i_loc == 0:
                fig_dict["data"] = [population_plot, candidates_plot]


    fig = go.Figure(fig_dict)
    fig.show()


if __name__ == '__main__':
    candidate_random_movement()
