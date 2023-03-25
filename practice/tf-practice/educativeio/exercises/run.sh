#!/bin/bash
terraform init
env TF_VAR_region=us-east-1
terraform apply