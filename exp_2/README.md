# Experiment 2: Impact of Synthetic Data Diversity on Model Safety

This directory contains the code and notebooks for Experiment 2, which investigates the impact of synthetic data diversity on the safety and adversarial robustness of fine-tuned models.

## Research Question

**RQ2**: Does increased diversity in synthetic data sources lead to better resistance to adversarial prompting in fine-tuned language models?

## Directory Structure

-   `harmful_response_generation/`: Scripts to generate responses from the fine-tuned models to harmful and adversarial prompts from the RefusalBench and ChatGPT Jailbreak Prompts datasets.
-   `quality_ratings/`: Scripts and notebooks to evaluate the quality of the generated responses.
-   `judge_llama/`: Contains the logic for the LLM-as-a-judge, which is used to evaluate the harmfulness of the generated responses.
-   `data_preprocessing.ipynb`: A notebook for preprocessing the prompts and responses before analysis.
-   `analysis_harmfulness.ipynb`: A notebook for analyzing the harmfulness ratings and generating the results for Experiment 2.

## Usage

1.  Run the scripts in `harmful_response_generation/` to generate responses to the adversarial prompts.
2.  Use `data_preprocessing.ipynb` to prepare the data for analysis.
3.  Use the `judge_llama/` scripts to get harmfulness ratings for the responses.
4.  Run `analysis_harmfulness.ipynb` and the notebooks in `quality_ratings/` to analyze the results. 