#!/bin/bash
aws s3 mb s3://cjf-terraform-state-bucket-123


aws dynamodb create-table --cli-input-json file://create-table-movies.json --endpoint-url http://localhost:8000
