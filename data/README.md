# Data Directory

This directory contains pre-collected data used for testing and development purposes.

## Structure

- `raw/`: Contains raw data files as collected from the source. There are two folders here
    - `metafile/`: The schema for this `csv` file collection is filepath, start_date, end_date. Where the filepath
    is the location of the logs for the failed tasks. start_date and end_date are also the metainformation of the failed tasks.
    - `logfile/`: The actual log files contents. As airflow stores logs in configured file location, we've collect log from those files.
- `processed/`: Contains processed data files that have been cleaned or transformed for use in the project. Here resides the files after bieng
    extracted and transformed for categorization.

## Usage

Place your pre-collected data files in the `raw/` directory. Use scripts or notebooks to process the data and save the results in the `processed/` directory.
