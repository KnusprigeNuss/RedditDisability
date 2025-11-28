
# RedditDisability – Topic Modeling & Analysis Pipeline
 

This project provides a full pipeline for collecting Reddit data, preprocessing it, training a BERTopic model, labeling discovered topics, and analyzing discussion patterns through both a web interface and command-line tools.

## Usage
To run the online tool run the following code and open the created URL:
```
cd tool
python3 menu.py
```

To run the complete pipeline in-terminal version run:
```
./run_pipeline.sh
```
Arguments can be used for this to decide between the search filters **top** (default), **hot** or **new** for the data collection and **manual** (default) or **automatic** for the topic labeling.
Example usage:
```
./run_pipeline.sh new manual
```

If you want to run singular steps of the pipeline you can use the run.sh file. To use it, you need to give the step you want to run as a flag. The flags are:

- ``` --collect ```, for data collection
      - with argument ``` top ``` (default), to fetch with the search filter top results
      - with argument ``` hot ```, to fetch with the search filter hot results 
      - with argument ``` new ```, to fetch with the search filter new results  
- ``` --preprocess ```, for the preprocessing of fetched data
- ``` --model ```, for the topic modeling 
- ``` --labeling ```, to label the found topics
       - with argument ``` manual ``` (default), to label the topics manually
       - with argument ``` automatic ```, to get automatically generated topic labels
- ``` --analysis ```, to generate data plots and engagement metrics 

Please make sure to only run one step at a time. Example usage:

```
./run.sh --collect top
```

## Repository Structure
Here I shortly explain what all the folders function is to avoid confusion. 
``` code ```-folder contains all python files that are required for the pipeline
``` data ```-folder is used by the files to save temporary data between pipeline steps
``` fetched_post ```-folder contains all posts that were fetched for this project
``` output ```-folder contains the output for the in-terminal pipeline (all figures and engagement table)
``` tool ```-folder contains all scripts and data required for the tool. This includes a intern data and figures folder 
``` trend_analysis ```-folder contains all scripts and data that were required for the trend analysis
``` trend_analysis_data ```-folder contains all posts that were fetched for the trend analysis

## Important Files

The project scope and written interpretation can be found in the root folder of the project.   
