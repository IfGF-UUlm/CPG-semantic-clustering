# Semantic Clustering of CPG recommendations
This GitHub repository contains the source code for the recommendation-clustering project, a tool for clustering and visualizing large collections of textual recommendations using the UMAP algorithm and KMeans clustering. The project was developed as part of the SURGE-Ahead (Supporting SURgery with GEriatric Co-Management and AI) project to create a digital healthcare application for assisting surgical teams in caring for geriatric patients.

## Project Overview
The recommendation-clustering project offers a scalable and efficient method for grouping similar recommendations and visualizing the clusters in 3D. This tool can be useful for analyzing and making sense of large collections of textual clinical practice guideline recommendations

## Project Structure
The repository has the following structure:
- [`LICENSE`](./LICENSE): MIT License for this project
- [`environment.yml`](./environment.yml): Requirements for setting up a virtual environment
- [`run.py`](./run.py): Main script for clustering and visualizing recommendations
- [`recommendations.csv`](./recommendations.csv): Sample dataset of recommendations used in the project

## Getting Started
To get started with recommendation-clustering, follow these steps:
Set up a virtual environment by running `conda env create -f environment.yml` and activate it
Run the script with `python3 run.py`
The resulting clusters will be saved in recommendations.csv and displayed in a 3D scatter plot, where each point represents a recommendation and is colored based on its cluster. The plot is saved as `AI_clusters.png` in the project directory.

## Dependencies
This project requires the following dependencies:
- pandas
- sentence-transformers
- umap
- scikit-learn
- matplotlib
You can install all dependencies by running `conda env create -f environment.yml`.

## License
This project is licensed under the MIT License.

## Citation
If you use this code in your research, please cite our paper: \
[The associated paper has been submitted for publication. The citation details will be updated once the paper is accepted.]

## Contact
For any questions, feedback, or concerns, please contact us at thomas.kocar@uni-ulm.de.
