-- Database initialization script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom types
CREATE TYPE error_severity AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE error_category AS ENUM (
    'resource_error',
    'data_quality',
    'dependency_failure',
    'configuration_error',
    'permission_error',
    'timeout_error',
    'unknown'
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_analysis_records_dag_task ON analysis_records(dag_id, task_id);
CREATE INDEX IF NOT EXISTS idx_analysis_records_timestamp ON analysis_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_analysis_records_category ON analysis_records(category);
CREATE INDEX IF NOT EXISTS idx_analysis_records_severity ON analysis_records(severity);

CREATE INDEX IF NOT EXISTS idx_baseline_records_dag_task ON baseline_records(dag_id, task_id);
CREATE INDEX IF NOT EXISTS idx_baseline_records_updated ON baseline_records(last_updated);

-- Create a view for daily statistics
CREATE OR REPLACE VIEW daily_stats AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_failures,
    COUNT(*) FILTER (WHERE severity = 'critical') as critical_failures,
    COUNT(*) FILTER (WHERE severity = 'high') as high_failures,
    COUNT(*) FILTER (WHERE severity = 'medium') as medium_failures,
    COUNT(*) FILTER (WHERE severity = 'low') as low_failures,
    AVG(processing_time) as avg_processing_time,
    AVG(confidence) as avg_confidence
FROM analysis_records
WHERE success = true
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Create a view for DAG statistics
CREATE OR REPLACE VIEW dag_stats AS
SELECT
    dag_id,
    COUNT(*) as total_failures,
    COUNT(DISTINCT task_id) as affected_tasks,
    AVG(processing_time) as avg_processing_time,
    MAX(timestamp) as last_failure,
    COUNT(*) FILTER (WHERE severity IN ('critical', 'high')) as severe_failures
FROM analysis_records
WHERE success = true
GROUP BY dag_id
ORDER BY total_failures DESC;
