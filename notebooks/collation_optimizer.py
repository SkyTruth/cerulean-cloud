import numpy as np
from tqdm.auto import tqdm
import plotly.graph_objects as go


class CollationOptimizer:
    """
    Optimizes collation by adjusting mean and standard deviation values for different source types.
    """

    def __init__(
        self,
        mean_std_dict,
        slick_source_dict,
        iter_params,
        ignore_lone_truth=False,
        target_weighting=None,
    ):
        """
        Initializes the optimizer with given mean/std values, data sources, and iteration parameters.

        Args:
            mean_std_dict (dict): Dictionary mapping source types to (mean, std) tuples.
            slick_source_dict (dict): Dictionary mapping source names to DataFrames.
            iter_params (list): List of iteration parameters for optimization.
            ignore_lone_truth (bool, optional): Whether to ignore single truth points without competition. Defaults to False.
        """
        self.mean_std_dict = mean_std_dict
        self.slick_source_dict = slick_source_dict
        self.targets = {}
        self.ignore_lone_truth = ignore_lone_truth
        self.scoring_method = self.score_by_bool_comparison
        self.footprint = {2: [], 3: []}
        self.target_over_iterations = []
        self.iter_params = iter_params

        self.target_weightings = (
            {
                "dark_slick_infra_source": 1.0,
                "dark_slick_vessel_source": 1.0,
                "infra_slick_vessel_source": 1.0,
                "infra_slick_dark_source": 1.0,
                "vessel_slick_infra_source": 1.0,
                "vessel_slick_dark_source": 1.0,
            }
            if target_weighting is None
            else target_weighting
        )

    def collate_dataframe(self, df, types=[1, 2, 3]):
        """
        Normalizes the coincidence scores within the DataFrame for specified source types.

        Args:
            df (pd.DataFrame): DataFrame containing collated data.
            types (list, optional): List of source types to normalize. Defaults to [1, 2, 3].
        """
        for t in types:
            t_mean, t_std = self.mean_std_dict[t]
            df.loc[df["source_type"] == t, "collated_score"] = (
                df[df["source_type"] == t]["coincidence_score"] - t_mean
            ) / t_std

    def collate_all_dataframes(self):
        """
        Applies collation normalization to all stored DataFrames.
        """
        for slick_source_df in self.slick_source_dict.keys():
            self.collate_dataframe(self.slick_source_dict[slick_source_df])

    def score_by_bool_comparison(self, g):
        """
        Scores a grouped DataFrame based on whether the highest-scoring prediction matches the truth.

        Args:
            g (pd.DataFrame): Grouped DataFrame of predictions.

        Returns:
            bool: Whether the highest-scoring prediction matches the ground truth.
        """
        truth = g[g["truth"]].iloc[0]
        top = g.sort_values(by="collated_score", ascending=False).iloc[0]
        return (
            top["source_type"] == truth["source_type"]
            or top["collated_score"] < truth["collated_score"]
        )

    def evaluate_collation(self, df):
        """
        Evaluates collation performance by comparing top-ranked predictions to the ground truth.

        Args:
            df (pd.DataFrame): DataFrame containing collated predictions.

        Returns:
            float: Proportion of correct top-ranked predictions.
        """
        truth_vs_top = []
        for _, g in df.groupby(by="slick_id"):
            if len(g) == 1 and self.ignore_lone_truth:
                continue
            if not g["truth"].any():
                continue
            truth_vs_top.append(g["truth"].iloc[g["collated_score"].argmax()])
        return sum(truth_vs_top) / len(truth_vs_top)

    def compute_collation_performance(self):
        """
        Computes overall collation performance across all data sources.

        Returns:
            float: Total collation performance score.
        """
        self.collate_all_dataframes()
        for slick_source_df in self.slick_source_dict.keys():
            self.targets[slick_source_df] = self.evaluate_collation(
                self.slick_source_dict[slick_source_df]
            )
        weighted_sum = sum(
            t * w
            for t, w in zip(self.targets.values(), self.target_weightings.values())
        )
        weighted_avg = weighted_sum / sum(self.target_weightings.values())
        # return sum(self.targets.values())
        return weighted_avg

    def compute_optimal_mean_std(
        self,
        fix_type,
        optimize_type,
        mean_range=[-2, 2],
        std_range=[0, 2],
        search_count=20,
    ):
        """
        Finds optimal mean and standard deviation values through grid search.

        Args:
            fix_type (int): Source type to keep fixed.
            optimize_type (int): Source type to optimize.
            mean_range (list, optional): Range of mean values to search. Defaults to [-2, 2].
            std_range (list, optional): Range of standard deviation values to search. Defaults to [0, 2].
            search_count (int, optional): Number of search steps. Defaults to 20.

        Returns:
            np.array: Best mean, std, and objective function score.
        """
        mean_values = np.linspace(mean_range[0], mean_range[1], num=search_count)
        std_values = np.linspace(std_range[0], std_range[1], num=search_count)
        obj_func_scores = []
        type_name = {2: "infra", 3: "dark"}
        for mean_adjustment in tqdm(
            mean_values,
            desc=f"Fixing {type_name[fix_type]} and optimizing {type_name[optimize_type]} mean and std...",
        ):
            for std_adjustment in std_values:
                self.mean_std_dict[optimize_type] = (mean_adjustment, std_adjustment)
                obj_func_scores.append(
                    [
                        mean_adjustment,
                        std_adjustment,
                        self.compute_collation_performance(),
                    ]
                )
        obj_func_scores = np.array(obj_func_scores)
        top_means_and_std = obj_func_scores[
            obj_func_scores[:, 2] == obj_func_scores[:, 2].max()
        ]
        print(f"{len(top_means_and_std)} optimal points found... choosing at random")
        return top_means_and_std[np.random.randint(0, len(top_means_and_std))]

    def perform_infra_dark_gradient_ascent(self):
        """
        Performs gradient ascent optimization over multiple iterations.
        """
        for i, params in enumerate(self.iter_params):
            print(f"ITERATION {i + 1} / {len(self.iter_params)}")
            print("Current MEAN/STD adjustments", self.mean_std_dict)
            m, s, t = self.compute_optimal_mean_std(**params)
            self.mean_std_dict[params["optimize_type"]] = (m, s)
            self.footprint[params["optimize_type"]].append([m, s, t])
            self.target_over_iterations.append(t)
        self.compute_collation_performance()
        print("Final MEAN/STD adjustments", self.mean_std_dict)

    def plot_target_over_iterations(self):
        """
        Plots optimization progress over iterations.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.target_over_iterations))),
                y=self.target_over_iterations,
                mode="lines+markers",
                name="Objective Function",
            )
        )
        fig.update_layout(
            title="Infra / Dark Gradient Ascent Optimization",
            xaxis_title="Iterations",
            yaxis_title="Objective Function",
            template="plotly_dark",
        )
        fig.show()
