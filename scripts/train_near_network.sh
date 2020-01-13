export TRANSFORM_TYPE="NEAR"
export N_EPOCHSa="20"
poetry run papermill ./train.ipynb output_near.ipynb
