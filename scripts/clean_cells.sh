poetry run jupyter nbconvert --to notebook --Exporter.preprocessors='["nbconvert.preprocessors.clearoutput.ClearOutputPreprocessor"]' $1
